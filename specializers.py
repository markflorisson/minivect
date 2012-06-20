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

        if not self.is_contig_specializer:
            for idx, arg in enumerate(node.arguments):
                if arg.is_array_funcarg:
                    stats.append(b.print_(b.constant("strides operand%d:" % idx),
                                          *self._index_list(arg.strides_pointer,
                                                            arg.type.ndim)))

        node.body = b.stats(b.stats(*stats), node.body)

    def visit_FunctionNode(self, node):
        b = self.astbuilder

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
        from minitypes import FunctionType, object_type
        b = self.astbuilder

        functype = FunctionType(return_type=object_type,
                                args=[object_type] * (2 + len(node.fmt_args)))
        return b.expr_stat(
            b.funccall(b.funcname(functype, "PyErr_Format"),
                       [node.exc_var, node.msg_val] + node.fmt_args))

    def visit_ErrorHandler(self, node):
        b = self.astbuilder

        node.error_variable = b.temp(minitypes.bool)
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

class StridedSpecializer(Specializer):

    specialization_name = "strided"

    order = "C"

    def visit_NDIterate(self, node):
        b = self.astbuilder

        self.indices = []
        node = node.body

        if self.order == "C":
            start = self.function.ndim - 1
            stop = -1
            step = -1
        else:
            start = 0
            stop = self.function.ndim
            step = 1

        for i in range(start, stop, step):
            upper = b.shape_index(i, self.function)
            node = b.for_range_upwards(node, upper=upper)
            self.indices.append(node.target)

        if self.order == "C":
            self.indices.reverse()

        return self.visit(node)

    def visit_Variable(self, node):
        if node.name in self.function.args and node.type.is_array:
            return self._element_location(node)

        return super(StridedSpecializer, self).visit_Variable(node)

    def _element_location(self, node):
        b = self.astbuilder
        ndim = node.type.ndim
        indices = [b.mul(index, b.stride(node, i))
                   for i, index in enumerate(self.indices[-ndim:])]
        pointer = b.cast(b.data_pointer(node),
                         minitypes.c_char_t.pointer())
        node = b.index_multiple(pointer, indices,
                                dest_pointer_type=node.type.dtype.pointer())
        self.visitchildren(node)
        return node

class StridedFortranSpecializer(StridedSpecializer):
    specialization_name = "strided_fortran"
    order = "F"

class ContigSpecializer(StridedSpecializer):

    specialization_name = "contig"
    is_contig_specializer = True

    def visit_FunctionNode(self, node):
        b = self.astbuilder

        # compute the product of the shape and insert it into the function body
        extents = [b.index(node.shape, b.constant(i))
                       for i in range(node.ndim)]
        node.total_shape = b.temp(node.shape.type.base_type)
        init_shape = b.assign(node.total_shape, reduce(b.mul, extents))
        node.body = b.stats(init_shape, node.body)

        return super(ContigSpecializer, self).visit_FunctionNode(node)

    def visit_NDIterate(self, node):
        node = self.astbuilder.for_range_upwards(
                        node.body, upper=self.function.total_shape)
        self.target = node.target
        self.visitchildren(node)
        return node

    def visit_StridePointer(self, node):
        return None

    def _element_location(self, node):
        data_pointer = self.astbuilder.data_pointer(node)
        return self.astbuilder.index(data_pointer, self.target)
