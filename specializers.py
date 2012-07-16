"""
Specializers for various sorts of data layouts and memory alignments.

These specializers operate on a copy of the simplified vector expression
representation (i.e., one with an NDIterate node). This node is replaced
with one or several ForNode nodes in a specialized order.

For auto-tuning code for tile size and OpenMP size, see
https://github.com/markflorisson88/cython/blob/_array_expressions/Cython/Utility/Vector.pyx
"""

import copy

import minivisitor
import miniutils
import minitypes

strength_reduction = True

class ASTMapper(minivisitor.VisitorTransform):
    """
    Base class to map foreign ASTs onto a minivect AST, or vice-versa.
    This sets the current node's position in the astbuilder for each
    node that is being visited, to make it easy to build new AST nodes
    without passing in source position information everywhere.
    """

    def __init__(self, context):
        super(ASTMapper, self).__init__(context)
        self.astbuilder = context.astbuilder

    def getpos(self, opaque_node):
        return self.context.getpos(opaque_node)

    def map_type(self, opaque_node, **kwds):
        "Return a mapped type for the foreign node."
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
    Base class for all specializers. Implement visit_* methods to specialize
    nodes to some pattern.

    Implements implementations to handle errors and cleanups, adds a return
    statement to the function and can insert debug print statements if
    context.debug is set to a true value.
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
        "Return a list of indexed pointers"
        return [self.astbuilder.index(pointer, self.astbuilder.constant(i))
                    for i in range(ndim)]

    def _debug_function_call(self, b, node):
        """
        Generate debug print statements when the specialized function is
        called.
        """
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
        """
        Handle a FunctionNode. Sets node.total_shape to the product of the
        shape, wraps the function's body in a
        :py:class:`minivect.miniast.ErrorHandler` if needed and adds a
        return statement.
        """
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
        """
        Replace with the setting of positional source information in case
        of an error.
        """
        b = self.astbuidler

        posinfo = self.function.posinfo
        if posinfo:
            pos = node.posinfo
            return b.stats(
                b.assign(b.deref(posinfo.filename), b.constant(pos.filename)),
                b.assign(b.deref(posinfo.lineno), b.constant(pos.lineno)),
                b.assign(b.deref(posinfo.column), b.constant(pos.column)))

    def visit_RaiseNode(self, node):
        """
        Generate a call to PyErr_Format() to set an exception.
        """
        from minitypes import FunctionType, object_
        b = self.astbuilder

        args = [object_] * (2 + len(node.fmt_args))
        functype = FunctionType(return_type=object_, args=args)
        return b.expr_stat(
            b.funccall(b.funcname(functype, "PyErr_Format"),
                       [node.exc_var, node.msg_val] + node.fmt_args))

    def get_type(self, type):
        "Resolve the type to the dtype of the array if an array type"
        if type.is_array:
            return type.dtype
        return type

    def visit_BinopNode(self, node):
        type = self.get_type(node.type)
        if node.operator == '%' and type.is_float:
            b = self.astbuilder
            functype = minitypes.FunctionType(return_type=type,
                                              args=[type, type])
            if type.itemsize == 4:
                modifier = "f"
            elif type.itemsize == 8:
                modifier = ""
            else:
                modifier = "l"

            fmod = b.variable(functype, "fmod%s" % modifier)
            return self.visit(b.funccall(fmod, [node.lhs, node.rhs]))

        self.visitchildren(node)
        return node

    def visit_ErrorHandler(self, node):
        """
        See miniast.ErrorHandler for an explanation of what this needs to do.
        """
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
        """
        Insert an OpenMP for loop with an 'if' clause that checks to see
        whether the total data size exceeds the given OpenMP auto-tuned size.
        The caller needs to adjust the size, set in the FunctionNode's
        'omp_size' attribute, depending on the number of computations.
        """
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
        node.total_shape = b.temp(node.shape.type.base_type)
        init_shape = b.assign(node.total_shape, reduce(b.mul, extents))
        node.body = b.stats(init_shape, node.body)
        return node.total_shape

    def loop_order(self, order, ndim=None):
        """
        Returns arguments to (x)range() to process something in C or Fortran
        order.
        """
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
        Return a ForNode ordered in C or Fortran order.
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
        """
        Process variables. For arrays, this means retrieving the element
        from the array through a call to self._element_location().
        """
        if node.name in self.function.args and node.type.is_array:
            return self._element_location(node)

        return super(OrderedSpecializer, self).visit_Variable(node)

    def _index_pointer(self, pointer, indices, strides):
        """
        Return an element for an N-dimensional index into a strided array.
        """
        b = self.astbuilder
        return b.index_multiple(
            b.cast(pointer, minitypes.char.pointer()),
            [b.mul(index, stride) for index, stride in zip(indices, strides)],
            dest_pointer_type=pointer.type)

    def _strided_element_location(self, node, indices=None, strides_index_offset=0,
                                  ndim=None):
        """
        Like _index_pointer, but given only an array operand indices. It first
        needs to get the data pointer and stride nodes.
        """
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
    """
    Specialize on the first or last dimension being contiguous (depending
    on the 'order' attribute).
    """

    specialization_name = "inner_contig_c"
    order = "C"

    def __init__(self, context, specialization_name=None):
        super(StridedCInnerContigSpecializer, self).__init__(context,
                                                             specialization_name)
        self.indices = []
        self.pointers = {}

    def _compute_inner_dim_pointer(self, arg, stats, tiled):
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

        indices = self.strided_indices()
        if tiled:
            # use all indices
            ndim = arg.type.ndim
            strides_offset = 0
            assert len(indices) == ndim, (indices, ndim)
            #indices = self.indices
        else:
            ndim = arg.type.ndim - 1
            strides_offset = self.order == 'F'
            #indices = self.strided_indices()

        first_element_pointer = self._strided_element_location(
                    arg, indices=indices, strides_index_offset=strides_offset,
                    ndim=ndim)

        stats.append(b.assign(pointer, first_element_pointer.operand))
        self.pointers[arg.variable] = pointer

    def computer_inner_dim_pointers(self, tiled=False):
        """
        Return a list of statements for temporary pointers we can directly
        index or add the final stride to.
        """
        stats = []
        for arg in self.function.arguments:
            if arg.is_array_funcarg:
                if arg.type.ndim >= 2:
                    self._compute_inner_dim_pointer(arg, stats, tiled)
                else:
                    self.pointers[arg.variable] = arg.data_pointer
                    arg.data_pointer.type = arg.data_pointer.type.unqualify("const")

        return stats

    def visit_NDIterate(self, node):
        """
        Replace this node with ordered loops and a direct index into a
        temporary data pointer in the contiguous dimension.
        """
        b = self.astbuilder
        # start by generating a C or Fortran ordered loop
        node = self.ordered_loop(node.body, self.indices)

        # get the second to last loop node, since its body is the
        # penultimate loop node
        loop = node

        stats = self.computer_inner_dim_pointers()
        if len(self.indices) > 1:
            for index in self.indices[:-2]:
                loop = node.body

            self.inner_loop = loop.body
            stats.append(b.pragma_for(self.inner_loop))
            loop.body = b.stats(*stats)
            node = self.omp_for(node)
        else:
            self.inner_loop = loop
            stats.append(self.omp_for(b.pragma_for(self.inner_loop)))
            node = b.stats(*stats)

        return self.visit(node)

    def strided_indices(self):
        "Return the list of strided indices for this order"
        return self.indices[:-1]

    def contig_index(self):
        "The contiguous index"
        return self.indices[-1]

    def _element_location(self, variable):
        data_pointer = self.pointers[variable]
        return self.astbuilder.index(data_pointer, self.contig_index())

class StridedFortranInnerContigSpecializer(StridedCInnerContigSpecializer):
    """
    Specialize on the first dimension being contiguous.
    """

    order = "F"
    specialization_name = "inner_contig_fortran"

    def strided_indices(self):
        return self.indices[1:]

    def contig_index(self):
        return self.indices[0]

class StrengthReducingStridedSpecializer(StridedCInnerContigSpecializer):
    """
    Specialize on strided operands. If some operands are contiguous in the
    dimension compatible with the order we are specializing for (the first
    if Fortran, the last if C), then perform a direct index into a temporary
    date pointer. For strided operands, perform strength reduction in the
    inner dimension by adding the stride to the data pointer in each iteration.
    """

    specialization_name = "strength_reduced_strided"
    order = "C"

    def matching_contiguity(self, type):
        """
        Check whether the array operand for the given type can be directly
        indexed.
        """
        return ((type.is_c_contig and self.order == "C") or
                (type.is_f_contig and self.order == "F"))

    def visit_NDIterate(self, node):
        b = self.astbuilder
        outer_loop = super(StridedSpecializer, self).visit_NDIterate(node)
        outer_loop = self.strength_reduce_inner_dimension(outer_loop,
                                                          self.inner_loop)
        return outer_loop

    def strength_reduce_inner_dimension(self, outer_loop, inner_loop):
        """
        Reduce the strength of strided array operands in the inner dimension,
        by adding the stride to the temporary pointer.
        """
        b = self.astbuilder

        outer_stats = []
        stats = []
        for arg in self.function.arguments:
            type = arg.variable.type
            if type is None:
                continue

            contig = self.matching_contiguity(type)
            if arg.variable in self.pointers and not contig:
                p = self.pointers[arg.variable]

                if self.order == "C":
                    inner_dim = type.ndim - 1
                else:
                    inner_dim = 0

                # Implement: temp_stride = strides[inner_dim] / sizeof(dtype)
                stride = b.stride(arg.variable, inner_dim)
                temp_stride = b.temp(stride.type.qualify("const"),
                                     name="temp_stride")
                outer_stats.append(
                    b.assign(temp_stride, b.div(stride, b.sizeof(type.dtype))))

                # Implement: temp_pointer += temp_stride
                stats.append(b.assign(p, b.add(p, temp_stride)))

        inner_loop.body = b.stats(inner_loop.body, *stats)
        outer_stats.append(outer_loop)
        return b.stats(*outer_stats)

    def _element_location(self, variable):
        """
        Generate a strided or directly indexed load of a single element.
        """
        if self.matching_contiguity(variable.type):
            # Generate a direct index in the pointer
            sup = super(StrengthReducingStridedSpecializer, self)
            return sup._element_location(variable)

        # strided access through temporary pointer
        return self.astbuilder.dereference(self.pointers[variable])

class StrengthReducingStridedFortranSpecializer(
    StridedFortranInnerContigSpecializer, StrengthReducingStridedSpecializer):
    """
    Specialize on Fortran order for strided operands and apply strength
    reduction in the inner dimension.
    """

    specialization_name = "strength_reduced_strided_fortran"
    order = "F"

class StridedSpecializer(StridedCInnerContigSpecializer):
    """
    Specialize on strided operands. If some operands are contiguous in the
    dimension compatible with the order we are specializing for (the first
    if Fortran, the last if C), then perform a direct index into a temporary
    date pointer.
    """

    specialization_name = "strided"
    order = "C"

    def matching_contiguity(self, type):
        """
        Check whether the array operand for the given type can be directly
        indexed.
        """
        return ((type.is_c_contig and self.order == "C") or
                (type.is_f_contig and self.order == "F"))

    def _compute_inner_dim_pointer(self, arg, stats):
        #if self.matching_contiguity(arg.type):
        super(StridedSpecializer, self)._compute_inner_dim_pointer(arg, stats)

    def _element_location(self, variable):
        """
        Generate a strided or directly indexed load of a single element.
        """
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
    """
    Specialize on Fortran order for strided operands.
    """

    specialization_name = "strided_fortran"
    order = "F"

if strength_reduction:
    StridedSpecializer = StrengthReducingStridedSpecializer
    StridedFortranSpecializer = StrengthReducingStridedFortranSpecializer

class ContigSpecializer(OrderedSpecializer):
    """
    Specialize on all specializations being contiguous (all F or all C).
    """

    specialization_name = "contig"
    is_contig_specializer = True

    def visit_NDIterate(self, node):
        """
        Generate a single ForNode over the total data size.
        """
        b = self.astbuilder
        for_node = b.for_range_upwards(node.body,
                                       upper=self.function.total_shape)
        node = self.omp_for(b.pragma_for(for_node))
        self.target = for_node.target
        return self.visit(node)

    def visit_StridePointer(self, node):
        return None

    def _element_location(self, node):
        "Directly index the data pointer"
        data_pointer = self.astbuilder.data_pointer(node)
        return self.astbuilder.index(data_pointer, self.target)

class CTiledStridedSpecializer(
    #StrengthReducingStridedSpecializer):
    StridedSpecializer):
    """
    Generate tiled code for the last two (C) or first two (F) dimensions.
    The blocksize may be overridden through the get_blocksize method, in
    a specializer subclass or mixin (see miniast.Context.specializer_mixin_cls).
    """
    specialization_name = "tiled_c"
    order = "C"
    is_tiled_specializer = True

    def get_blocksize(self):
        """
        Get the tile size. Override in subclasses to provide e.g. parametric
        tiling.
        """
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

        # Generate the two outer tiling loops
        tiled_loop_body = b.stats(b.constant(0)) # fake empty loop body
        body = self.ordered_loop(tiled_loop_body, self.tiled_indices,
                                 step=self.blocksize,
                                 loop_order=self.tiled_order())
        del tiled_loop_body.stats[:]

        # Generate some temporaries to store the upper limit of the inner
        # tiled loops
        upper_limits = {}
        stats = []
        # sort the indices in forward order, to match up with the ordered
        # indices
        tiled_order = sorted(range(*self.tiled_order()))
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

        # Generate the inner tiled loops
        outer_for_node = node.body
        inner_body = node.body

        inner_loops = self.ordered_loop(
            node.body, self.indices,
            lower=lower, upper=upper,
            loop_order=self.tiled_order())

        tiled_loop_body.stats.append(inner_loops)
        innermost_loop = inner_loops.body

        # Generate the outer loops (in case the array operands have more than
        # two dimensions)
        indices = []
        body = self.ordered_loop(body, indices,
                                 loop_order=self.untiled_order())

        body = self.omp_for(body)
        # At this point, 'self.indices' are the indices of the tiled loop
        # (the indices in the first two dimensions for Fortran,
        #  the indices in the last two # dimensions for C)
        # 'indices' are the indices of the outer loops
        if self.order == "C":
            self.indices = indices + self.indices
        else:
            self.indices = self.indices + indices

        # Generate temporary pointers for all operands and insert just outside
        # the innermost loop
        stats = self.computer_inner_dim_pointers(tiled=True)
        stats.append(b.pragma_for(inner_loops.body))
        inner_loops.body = b.stats(*stats)

        if strength_reduction:
            body = self.strength_reduce_inner_dimension(body, innermost_loop)

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

    def strided_indices(self):
        return self.indices[:-1] + [self.tiled_indices[1]]

    # def _element_location(self, variable):
    #     return self._strided_element_location(variable)

class FTiledStridedSpecializer(StridedFortranSpecializer,
                               #StrengthReducingStridedFortranSpecializer,
                               CTiledStridedSpecializer):
    "Tile in Fortran order"

    specialization_name = "tiled_fortran"
    order = "F"

    def tiled_order(self):
        "Tile in the first two dimensions"
        return 0, 2, 1

    def untiled_order(self):
        return 2, self.function.ndim, 1

    def strided_indices(self):
        return [self.tiled_indices[0]] + self.indices[1:]