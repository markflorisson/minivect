"""
Specializers for various sorts of data layouts and memory alignments.
"""

import copy

import minivisitor
import miniutils
import minitypes

class ASTMapper(minivisitor.VisitorTransform):

    def __init__(self, context):
        super(ASTMapper, self).__init__(context)
        self.astbuilder = context.astbuilder

    def getpos(self, opaque_node):
        return self.context.getpos(opaque_node)

    def map_type(self, opaque_node, **kwds):
        return self.context.typemapper.map_type(
                        self.context.gettype(opaque_node), **kwds)

    def visit(self, node, *args):
        prev = self.astbuilder.pos
        self.astbuilder.pos = node.pos
        result = super(ASTMapper, self).visit(node)
        self.astbuilder.pos = prev
        return result

class Specializer(ASTMapper):
    """
    Implement visit_* methods to specialize to some pattern. The default is
    to copy the node and specialize the children.
    """

    is_contig_specializer = False
    is_tiled_specializer = False

    def __init__(self, context, specialization_name=None):
        super(Specializer, self).__init__(context)
        if specialization_name is not None:
            self.specialization_name = specialization_name

        self.variables = {}
        self.error_handlers = []

    def getpos(self, node):
        return node.pos

    def visit(self, node, *args):
        result = super(Specializer, self).visit(node)
        if result is not None:
            result.is_specialized = True
        return result

    def visit_Node(self, node):
        # node = copy.copy(node)
        self.visitchildren(node)
        return node

    def _index_list(self, pointer, ndim):
        return [self.astbuilder.index(pointer, self.astbuilder.constant(i))
                    for i in range(ndim)]

    def _debug_function_call(self, b, node):
        stats = [
            b.print_(b.constant("Calling function %s (%s specializer)" % (
                                       node.name, self.specialization_name))),
            b.print_(b.constant("shape:"), *self._index_list(node.shape,
                                                             node.ndim)),
        ]
        if self.is_tiled_specializer:
            stats.append(b.print_(b.constant("blocksize:"), self.get_blocksize()))

        if not self.is_contig_specializer:
            for idx, arg in enumerate(node.arguments):
                if arg.is_array_funcarg:
                    stats.append(b.print_(b.constant("strides operand%d:" % idx),
                                          *self._index_list(arg.strides_pointer,
                                                            arg.type.ndim)))
                    stats.append(b.print_(b.constant("data pointer %d:" % idx),
                                          arg.data_pointer))

        node.body = b.stats(b.stats(*stats), node.body)

    def visit_FunctionNode(self, node):
        b = self.astbuilder
        self.compute_total_shape(node)

        # set this so bad people can specialize during code generation time
        node.specializer = self
        node.specialization_name = self.specialization_name
        self.function = node

        if self.context.debug:
            self._debug_function_call(b, node)

        if node.body.may_error(self.context):
            node.body = b.error_handler(node.body)

        node.body = b.stats(node.body, b.return_(node.success_value))

        self.visitchildren(node)
        return node

    def visit_ForNode(self, node):
        if node.body.may_error(self.context):
            node.body = self.astbuilder.error_handler(node.body)
        self.visitchildren(node)
        return node

    def visit_Variable(self, node):
        if node.name not in self.variables:
            self.variables[node.name] = node
        return self.visit_Node(node)

    def visit_PositionInfoNode(self, node):
        b = self.astbuidler

        posinfo = self.function.posinfo
        if posinfo:
            pos = node.posinfo
            return b.stats(
                b.assign(b.deref(posinfo.filename), b.constant(pos.filename)),
                b.assign(b.deref(posinfo.lineno), b.constant(pos.lineno)),
                b.assign(b.deref(posinfo.column), b.constant(pos.column)))

    def visit_RaiseNode(self, node):
        from minitypes import FunctionType, object_
        b = self.astbuilder

        args = [object_] * (2 + len(node.fmt_args))
        functype = FunctionType(return_type=object_, args=args)
        return b.expr_stat(
            b.funccall(b.funcname(functype, "PyErr_Format"),
                       [node.exc_var, node.msg_val] + node.fmt_args))

    def visit_ErrorHandler(self, node):
        b = self.astbuilder

        node.error_variable = b.temp(minitypes.bool_)
        node.error_var_init = b.assign(node.error_variable, 0)
        node.cleanup_jump = b.jump(node.cleanup_label)
        node.error_target_label = b.jump_target(node.error_label)
        node.cleanup_target_label = b.jump_target(node.cleanup_label)
        node.error_set = b.assign(node.error_variable, 1)

        if self.error_handlers:
            cascade_code = b.jump(self.error_handlers[-1].error_label)
        else:
            cascade_code = b.return_(self.function.error_value)

        node.cascade = b.if_(node.error_variable, cascade_code)

        self.error_handlers.append(node)
        self.visitchildren(node)
        self.error_handlers.pop()
        return node

    def omp_for(self, node):
        if_clause = self.astbuilder.binop(minitypes.bool_, '>',
                                          self.function.total_shape,
                                          self.function.omp_size)
        return self.astbuilder.omp_for(node, if_clause)

class OrderedSpecializer(Specializer):
    """
    Specializer that understands C and Fortran data layout orders.
    """

    def compute_total_shape(self, node):
        """
        Compute the product of the shape (entire length of array output).
        Sets the total shape as attribute of the function (total_shape).
        """
        b = self.astbuilder
        # compute the product of the shape and insert it into the function body
        extents = [b.index(node.shape, b.constant(i))
                   for i in range(node.ndim)]
        node.total_shape = b.temp(node.shape.type.base_type.unqualify("const"))
        init_shape = b.assign(node.total_shape, reduce(b.mul, extents))
        node.body = b.stats(init_shape, node.body)
        return node.total_shape

    def loop_order(self, order, ndim=None):
        if ndim is None:
            ndim = self.function.ndim

        if order == "C":
            return self.c_loop_order(ndim)
        else:
            return self.f_loop_order(ndim)

    def c_loop_order(self, ndim):
        return ndim - 1, -1, -1

    def f_loop_order(self, ndim):
        return 0, ndim, 1

    def order_indices(self, indices):
        """
        Put the indices of the for loops in the right iteration order. The
        loops were build backwards (Fortran order), so for C we need to
        reverse them.

        Note: the indices are always ordered on the dimension they index
        """
        if self.order == "C":
            indices.reverse()

    def ordered_loop(self, node, result_indices, lower=None, upper=None,
                     step=None, loop_order=None):
        """
        Return a loop ordered in C or Fortran order.
        """
        b = self.astbuilder

        if lower is None:
            lower = lambda i: None
        if upper is None:
            upper = lambda i: b.shape_index(i, self.function)
        if loop_order is None:
            loop_order = self.loop_order(self.order)

        indices = []
        # print range(*self.loop_order(self.order))
        for i in range(*loop_order):
            node = b.for_range_upwards(node, lower=lower(i), upper=upper(i),
                                       step=step)
            indices.append(node.target)

        self.order_indices(indices)
        result_indices.extend(indices)
        return node

    def visit_Variable(self, node):
        if node.name in self.function.args and node.type.is_array:
            return self._element_location(node)

        return super(OrderedSpecializer, self).visit_Variable(node)

    def _index_pointer(self, pointer, indices, strides):
        b = self.astbuilder
        return b.index_multiple(
            b.cast(pointer, minitypes.char.pointer()),
            [b.mul(index, stride) for index, stride in zip(indices, strides)],
            dest_pointer_type=pointer.type)

    def _strided_element_location(self, node, indices=None, strides_index_offset=0,
                                  ndim=None):
        indices = indices or self.indices
        b = self.astbuilder
        if ndim is None:
            ndim = node.type.ndim

        indices = [index for index in indices[len(indices) - ndim:]]
        strides = [b.stride(node, i + strides_index_offset)
                       for i, idx in enumerate(indices)]
        node = self._index_pointer(b.data_pointer(node), indices, strides)
        self.visitchildren(node)
        return node

class StridedCInnerContigSpecializer(OrderedSpecializer):

    specialization_name = "inner_contig_c"
    order = "C"

    def __init__(self, context, specialization_name=None):
        super(StridedCInnerContigSpecializer, self).__init__(context,
                                                             specialization_name)
        self.indices = []
        self.pointers = {}

    def _compute_inner_dim_pointer(self, arg, stats):
        """
        Compute the pointer to each 'row'.

        In the case of Fortran, offset strides by one, since we want all
        strides but the first.
        In the case of C, we want all strides except the last.

        (indices is already offset through the strided_indices()
        method)

        arg: the array function argument
        stats: list of statements we append to
        """
        b = self.astbuilder

        dest_pointer_type = arg.data_pointer.type.unqualify('const')
        pointer = b.temp(dest_pointer_type)
        first_element_pointer = self._strided_element_location(
            arg, indices=self.strided_indices(),
            strides_index_offset=self.order == 'F',
            ndim=arg.type.ndim - 1)
        stats.append(b.assign(pointer, first_element_pointer.operand))
        self.pointers[arg.variable] = pointer

    def visit_NDIterate(self, node):
        b = self.astbuilder
        # start by generating a C or Fortran ordered loop
        node = self.ordered_loop(node.body, self.indices)

        # get the second to last loop node, since its body is the
        # penultimate loop node
        loop = node
        for index in self.indices[:-2]:
            loop = node.body

        stats = []
        for arg in self.function.arguments:
            if arg.is_array_funcarg:
                self._compute_inner_dim_pointer(arg, stats)

        loop.body = b.stats(*(stats + [loop.body]))
        return self.visit(self.omp_for(node))

    def strided_indices(self):
        return self.indices[:-1]

    def contig_index(self):
        return self.indices[-1]

    def _element_location(self, variable):
        data_pointer = self.pointers[variable]
        return self.astbuilder.index(data_pointer, self.contig_index())

class StridedFortranInnerContigSpecializer(StridedCInnerContigSpecializer):

    order = "F"
    specialization_name = "inner_contig_fortran"

    def strided_indices(self):
        return self.indices[1:]

    def contig_index(self):
        return self.indices[0]

class StridedSpecializer(StridedCInnerContigSpecializer):

    specialization_name = "strided"
    order = "C"

    def matching_contiguity(self, type):
        return ((type.is_c_contig and self.order == "C") or
                (type.is_f_contig and self.order == "F"))

    def _compute_inner_dim_pointer(self, arg, stats):
        #if self.matching_contiguity(arg.type):
        super(StridedSpecializer, self)._compute_inner_dim_pointer(arg, stats)

    def _element_location(self, variable):
        #if variable in self.pointers:
        if self.matching_contiguity(variable.type):
            return super(StridedSpecializer, self)._element_location(variable)

        b = self.astbuilder
        pointer = self.pointers[variable]
        indices = [self.contig_index()]

        if self.order == "C":
            inner_dim = variable.type.ndim - 1
        else:
            inner_dim = 0

        strides = [b.stride(variable, inner_dim)]
        return self._index_pointer(pointer, indices, strides)

class StridedFortranSpecializer(StridedFortranInnerContigSpecializer,
                                StridedSpecializer):
    specialization_name = "strided_fortran"
    order = "F"

class ContigSpecializer(OrderedSpecializer):

    specialization_name = "contig"
    is_contig_specializer = True

    def visit_NDIterate(self, node):
        b = self.astbuilder
        node = self.omp_for(b.for_range_upwards(
                    node.body, upper=self.function.total_shape))
        self.target = node.for_node.target
        return self.visit(node)

    def visit_StridePointer(self, node):
        return None

    def _element_location(self, node):
        data_pointer = self.astbuilder.data_pointer(node)
        return self.astbuilder.index(data_pointer, self.target)

class CTiledStridedSpecializer(OrderedSpecializer):

    specialization_name = "tiled_c"
    order = "C"
    is_tiled_specializer = True

    def get_blocksize(self):
        return self.astbuilder.constant(128)

    def tiled_order(self):
        "Tile in the last two dimensions"
        return self.function.ndim - 1, self.function.ndim - 1 - 2, -1

    def untiled_order(self):
        return self.function.ndim - 1 - 2, -1, -1

    def visit_NDIterate(self, node):
        return self._tile_in_two_dimensions(node)

    def _tile_in_two_dimensions(self, node):
        """
        This version generates tiling loops in the first or last two dimensions
        (depending on C or Fortran order).
        """
        b = self.astbuilder

        self.tiled_indices = []
        self.indices = []
        self.blocksize = self.get_blocksize()

        tiled_loop_body = b.stats(b.constant(0)) # fake empty loop body
        body = self.ordered_loop(tiled_loop_body, self.tiled_indices,
                                 step=self.blocksize,
                                 loop_order=self.tiled_order())
        body = self.omp_for(body)
        del tiled_loop_body.stats[:]

        upper_limits = {}
        stats = []
        tiled_order = range(*self.tiled_order())
        for i, index in zip(tiled_order, self.tiled_indices):
            upper_limit = b.temp(index.type)
            tiled_loop_body.stats.append(
                b.assign(upper_limit, b.min(b.add(index, self.blocksize),
                                            b.shape_index(i, self.function))))
            upper_limits[i] = upper_limit

        tiled_indices = dict(zip(tiled_order, self.tiled_indices))
        def lower(i):
            if i in tiled_indices:
                return tiled_indices[i]
            return None

        def upper(i):
            if i in upper_limits:
                return upper_limits[i]
            return b.shape_index(i, self.function)

        outer_for_node = node.body
        inner_body = node.body
        tiled_loop_body.stats.append(self.ordered_loop(
                node.body, self.indices,
                lower=lower, upper=upper,
                loop_order=self.tiled_order()))

        indices = []
        body = self.ordered_loop(body, indices,
                                 loop_order=self.untiled_order())

        # At this point, 'self.indices' are the indices of the tiled loop
        # (the indices in the first two dimensions for Fortran,
        #  the indices in the last two # dimensions for C)
        # 'indices' are the indices of the outer loops
        if self.order == "C":
            self.indices = indices + self.indices
        else:
            self.indices = self.indices + indices

        return self.visit(body)

    def _tile_in_all_dimensions(self, node):
        """
        This version generates tiling loops in all dimensions.
        """
        b = self.astbuilder

        self.tiled_indices = []
        self.indices = []
        self.blocksize = self.get_blocksize()

        tiled_loop_body = b.stats(b.constant(0)) # fake empty loop body
        body = self.ordered_loop(tiled_loop_body, self.tiled_indices,
                                 step=self.blocksize)
        body = self.omp_for(body)
        del tiled_loop_body.stats[:]

        upper_limits = []
        stats = []
        for i, index in enumerate(self.tiled_indices):
            upper_limit = b.temp(index.type)
            tiled_loop_body.stats.append(
                b.assign(upper_limit, b.min(b.add(index, self.blocksize),
                                            b.shape_index(i, self.function))))
            upper_limits.append(upper_limit)

        tiled_loop_body.stats.append(self.ordered_loop(
            node.body, self.indices,
            lower=lambda i: self.tiled_indices[i],
            upper=lambda i: upper_limits[i]))
        return self.visit(body)

    def _element_location(self, variable):
        return self._strided_element_location(variable)

class FTiledStridedSpecializer(CTiledStridedSpecializer):

    specialization_name = "tiled_fortran"
    order = "F"

    def tiled_order(self):
        "Tile in the first two dimensions"
        return 0, 2, 1

    def untiled_order(self):
        return 2, self.function.ndim, 1