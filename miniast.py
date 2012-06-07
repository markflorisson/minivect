"""
This module provides the AST. Subclass Context and override the various
methods to allow minivect visitors over the AST, to promote and map types,
etc. Subclass and override ASTBuilder's method to provide alternative
AST nodes or different implementations.
"""

import copy
import types

import minitypes
import miniutils
import minivisitor
import minicode
import codegen

class Context(object):
    """
    A context that knows how to map ASTs back and forth, how to wrap nodes
    and types, and how to instantiate a code generator for specialization.

    An opaque_node is a node that is not from our AST, and a normal node
    is one that has a interface compatible with ours.
    """

    codegen_cls = codegen.CodeGen
    cleanup_codegen_cls = codegen.CodeGenCleanup
    codewriter_cls = minicode.CodeWriter
    codeformatter_cls = minicode.CodeFormatter

    specializer_mixin_cls = None

    def __init__(self, astbuilder=None, typemapper=None):
        self.astbuilder = astbuilder or ASTBuilder(self)
        self.typemapper = typemapper or minitypes.TypeMapper()

    def run_opaque(self, astmapper, opaque_ast, specializers):
        return self.run(astmapper.visit(opaque_ast), specializers)

    def run(self, ast, specializer_classes):
        for specializer_class in specializer_classes:
            if self.specializer_mixin_cls:
                cls1, cls2 = specializer_class, self.specializer_mixin_cls
                name = "%s_%s" % (cls1.__name__, cls2.__name__)
                specializer_class = type(name, (cls1, cls2), {})

            specializer = specializer_class(self)
            specialized_ast = specializer.visit(copy.deepcopy(ast))
            # specialized_ast.print_tree(self)
            codewriter = self.codewriter_cls(self)
            visitor = self.codegen_cls(self, codewriter)
            visitor.visit(specialized_ast)
            yield (specializer, specialized_ast, codewriter,
                   self.codeformatter_cls().format(codewriter))

    def generate_disposal_code(self, code, node):
        transform = self.cleanup_codegen_cls(self, code)
        transform.visit(node)

    #
    ### Override in subclasses where needed
    #

    def promote_types(self, type1, type2):
        return minitypes.promote(type1, type2)

    def getchildren(self, node):
        "Implement to allow a minivisitor.Visitor over a foreign AST."
        return node.child_attrs

    def getpos(self, opaque_node):
        "Get the position of a foreign node"
        filename, line, col = opaque_node.pos
        return Position(filename, line, col)

    def gettype(self, opaque_node):
        "Get a type of a foreign node"
        return opaque_node.type

    def may_error(self, opaque_node):
        "Return whether this node may result in an exception."
        raise NotImplementedError

    def declare_type(self, type):
        "Return a declaration for a type"
        return type.tostring(self)

class CContext(Context):

    codegen_cls = codegen.CCodeGen
    codewriter_cls = minicode.CCodeWriter
    codeformatter_cls = minicode.CCodeStringFormatter

class ASTBuilder(object):
    """
    This class is used to build up ASTs. It can be used by a user from
    a transform or otherwise, but the important bit is that we use it
    in our code to build up an AST that can be overridden by the user.
    """

    # the 'pos' attribute is set for each visit to each node by
    # the ASTMapper
    pos = None
    tempcounter = 0

    def __init__(self, context):
        self.context = context

    def infer_type(self, value):
        if isinstance(value, (int, long)):
            return minitypes.IntType()
        elif isinstance(value, float):
            return minitypes.FloatType()
        elif isinstance(value, str):
            return minitypes.CStringType()
        else:
            raise minierror.InferTypeError()

    def function(self, name, body, arguments, shapevar, posinfo=None):
        """
        arguments: [FunctionArgument]
        shapevar: the shape Variable. Will be prepended as
                  an argument to `arguments`
        """
        arguments.insert(0, self.funcarg(shapevar))
        arguments.insert(1, posinfo)
        body = NDIterate(self.pos, body)
        return FunctionNode(self.pos, name, body, arguments, shapevar,
                            posinfo,
                            error_value=self.constant(-1),
                            success_value=self.constant(0))

    def funcarg(self, variable, *variables):
        if not variables:
            variables = [variable]
        return FunctionArgument(self.pos, variable, list(variables))

    def array_funcarg(self, variable):
        return ArrayFunctionArgument(
                self.pos, variable.type, name=variable.name,
                variable=variable,
                data_pointer=self.data_pointer(variable),
                #shape_pointer=self.shapevar(variable),
                strides_pointer=self.stridesvar(variable))

    def funccall(self, name_or_pointer, args):
        type = name_or_pointer.type
        if name_or_pointer.is_pointer:
            type = name_or_pointer.type.base_type
        return FuncCallNode(self.pos, type=type,
                            name_or_pointer=name_or_pointer, args=args)

    def funcname(self, type, name):
        return FuncNameNode(self.pos, type=type, name=name)

    def funcref(self, function):
        return FuncRefNode(self.pos, type=function.type, function=function)

    def for_(self, body, init, condition, step, is_tiled=False):
        return ForNode(self.pos, init, condition, step, body, is_tiled)

    def for_range_upwards(self, body, upper, lower=None):
        if lower is None:
            lower = self.constant(0)

        temp = self.temp(minitypes.Py_ssize_t)
        init = self.assign_expr(temp, lower)
        condition = self.binop(minitypes.bool, '<', temp, upper)
        step = self.assign_expr(temp, self.add(temp, self.constant(1)))

        result = self.for_(body, init, condition, step)
        result.target = temp
        return result

    def stats(self, *statements):
        stats = []
        for stat in statements:
            if stat.is_statlist:
                stats.extend(stat.stats)
            else:
                stats.append(stat)

        return StatListNode(self.pos, stats)

    def expr_stat(self, expr):
        return ExprStatNode(expr.pos, expr=expr)

    def if_(self, cond, body):
        return IfNode(self.pos, cond=cond, body=body)

    def binop(self, type, op, lhs, rhs):
        return BinopNode(self.pos, type, op, lhs, rhs)

    def add(self, lhs, rhs):
        if lhs.is_constant and lhs.value == 0:
            return rhs
        elif rhs.is_constant and rhs.value == 0:
            return lhs

        type = self.context.promote_types(lhs.type, rhs.type)
        return self.binop(type, '+', lhs, rhs)

    def mul(self, lhs, rhs):
        if lhs.is_constant and lhs.value == 1:
            return rhs
        elif rhs.is_constant and rhs.value == 1:
            return lhs

        type = self.context.promote_types(lhs.type, rhs.type)
        return self.binop(type, '*', lhs, rhs)

    def index(self, pointer, index, dest_pointer_type=None):
        if dest_pointer_type:
            return self.index_multiple(pointer, [index], dest_pointer_type)
        return SingleIndexNode(self.pos, pointer.type.base_type,
                               pointer, index)

    def index_multiple(self, pointer, indices, dest_pointer_type=None):
        for index in indices:
            pointer = self.add(pointer, index)

        if dest_pointer_type is not None:
            pointer = self.cast(pointer, dest_pointer_type)

        return self.dereference(pointer)

    def assign_expr(self, node, value):
        assert node is not None
        if not isinstance(value, Node):
            value = self.constant(value)
        return AssignmentExpr(self.pos, node.type, node, value)

    def assign(self, node, value):
        return self.expr_stat(self.assign_expr(node, value))

    def dereference(self, pointer):
        return DereferenceNode(self.pos, pointer.type.base_type, pointer)

    def unop(self, op, operand):
        return UnopNode(self.pos, op, operand)

    def temp(self, type):
        self.tempcounter += 1
        return TempNode(self.pos, type, 'temp%d' % self.tempcounter)

    def constant(self, value, type=None):
        if type is None:
            type = self.infer_type(value)
        return ConstantNode(self.pos, type, value)

    def variable(self, type, name):
        return Variable(self.pos, type, name)

    def cast(self, node, dest_type):
        return CastNode(self.pos, dest_type, node)

    def return_(self, result):
        return ReturnNode(self.pos, result)

    def data_pointer(self, variable):
        assert variable.type.is_array
        return DataPointer(self.pos, variable.type.dtype.pointer(),
                           variable)

    def shape_index(self, index, function):
        return self.index(function.shape, self.constant(index))

    def extent(self, variable, index, function):
        "Index the shape of a specific variable"
        assert variable.type.is_array
        offset = function.ndim - variable.type.ndim
        return self.index(function.shape, self.constant(index + offset))

    def stridesvar(self, variable):
        return StridePointer(self.pos, minitypes.Py_ssize_t.pointer(), variable)

    def stride(self, variable, index):
        return self.index(self.stridesvar(variable), self.constant(index))

    def jump(self, label):
        return JumpNode(self.pos, label)

    def jump_target(self, label):
        return JumpTargetNode(self.pos, label)

    def label(self, name):
        return LabelNode(self.pos, name)

    def raise_exc(self, posinfo, exc_var, msg_val, fmt_args):
        return RaiseNode(self.pos, posinfo, exc_var, msg_val, fmt_args)

    def posinfo(self, posvars):
        return PositionInfoNode(self.pos, posinfo=posvars)

    def error_handler(self, node):
        return ErrorHandler(self.pos, body=node,
                            error_label=self.label('error'),
                            cleanup_label=self.label('cleanup'))

    def wrap(self, opaque_node, specialize_node_callback, **kwds):
        type = minitypes.TypeWrapper(self.context.gettype(opaque_node))
        return NodeWrapper(self.context.getpos(opaque_node), type,
                           opaque_node, specialize_node_callback, **kwds)

class Position(object):
    def __init__(self, filename, line, col):
        self.filename = filename
        self.line = line
        self.col = col

    def __str__(self):
        return "%s:%d:%d" % (self.filename, self.line, self.col)

class Node(miniutils.ComparableObjectMixin):

    is_expression = False

    is_statlist = False
    is_scalar = False
    is_constant = False
    is_assignment = False
    is_unop = False
    is_binop = False

    is_node_wrapper = False
    is_data_pointer = False
    is_jump = False
    is_label = False

    is_funcarg = False
    is_array_funcarg = False

    is_specialized = False

    child_attrs = []

    def __init__(self, pos, **kwds):
        self.pos = pos
        vars(self).update(kwds)

    def may_error(self, context):
        """
        Return whether something may go wrong and we need to jump to an
        error handler.
        """
        visitor = minivisitor.MayErrorVisitor(context)
        visitor.visit(self)
        return visitor.may_error

    def print_tree(self, context):
        visitor = minivisitor.PrintTree(context)
        visitor.visit(self)

    @property
    def comparison_objects(self):
        type = getattr(self, 'type', None)
        if type is None:
            return self.children
        return tuple(self.children) + (type,)

    def __eq__(self, other):
        # Don't use isinstance here, compare on exact type to be consistent
        # with __hash__. Override where sensible
        return (type(self) is type(other) and
                self.comparison_objects == other.comparison_objects)

    def __hash__(self):
        h = hash(type(self))
        for subtype in self.comparison_type_list:
            h = h ^ hash(subtype)

        return h

class ExprNode(Node):
    is_expression = True

    def __init__(self, pos, type, **kwds):
        super(ExprNode, self).__init__(pos, **kwds)
        self.type = type

class FunctionNode(Node):
    child_attrs = ['body', 'arguments']
    def __init__(self, pos, name, body, arguments, shape, posinfo,
                 error_value, success_value):
        super(FunctionNode, self).__init__(pos)
        self.name = name
        self.body = body
        self.arguments = arguments
        self.shape = shape
        self.posinfo = posinfo
        self.error_value = error_value
        self.success_value = success_value

        self.args = dict((v.name, v) for v in arguments)
        self.ndim = max(arg.type.ndim for arg in arguments
                                          if arg.type and arg.type.is_array)


class FuncCallNode(Node):
    """
    Call a function given a pointer or its name (FuncNameNode)
    """
    child_attrs = ['name_or_pointer', 'args']

class FuncNameNode(Node):
    """
    Load a function given its C linkage name.
    """

class FuncRefNode(Node):
    """
    Refer to a function given a FunctionNode.
    """

class ReturnNode(Node):
    child_attrs = ['operand']
    def __init__(self, pos, operand):
        super(ReturnNode, self).__init__(pos)
        self.operand = operand

class RaiseNode(Node):
    "Raise a Python exception. The callee must hold the GIL."

    child_attrs = ['posinfo', 'exc_var', 'msg_val', 'fmt_args']

    def __init__(self, pos, posinfo, exc_var, msg_val, fmt_args):
        super(RaiseNode, self).__init__(pos)
        self.posinfo = posinfo
        self.exc_var, self.msg_val, self.fmt_args = (exc_var, msg_val, fmt_args)

class PositionInfoNode(Node):
    """
    Node that holds a position of where an error occurred. This position
    needs to be returned to the callee if the callee supports it.
    """

class FunctionArgument(ExprNode):
    """
    Argument to the FunctionNode. Array arguments contain multiple
    actual arguments, e.g. the data and stride pointer.

        variable: some argument to the function (array or otherwise)
        variables: the actual variables this operand should be unpacked into
    """
    child_attrs = ['variables']
    if_funcarg = True

    def __init__(self, pos, variable, variables):
        super(FunctionArgument, self).__init__(pos, variable.type)
        self.variables = variables
        self.variable = variable
        self.name = variable.name
        self.args = dict((v.name, v) for v in variables)

class ArrayFunctionArgument(ExprNode):
    child_attrs = ['data_pointer', 'strides_pointer']
    is_array_funcarg = True

class NDIterate(Node):

    child_attrs = ['body']

    def __init__(self, pos, body):
        super(NDIterate, self).__init__(pos)
        self.body = body

class ForNode(Node):

    child_attrs = ['init', 'condition', 'step', 'body']

    def __init__(self, pos, init, condition, step, body, is_tiled=False):
        """
        init, condition and step are the 3 arguments to the supposed
        C for loop
        """
        super(ForNode, self).__init__(pos)
        self.init = init
        self.condition = condition
        self.step = step
        self.body = body
        self.is_tiled = False

class IfNode(Node):
    child_attrs = ['cond', 'body']

class StatListNode(Node):
    child_attrs = ['stats']
    is_statlist = True

    def __init__(self, pos, statements):
        super(StatListNode, self).__init__(pos)
        self.stats = statements

class ExprStatNode(Node):
    child_attrs = ['expr']

class NodeWrapper(ExprNode):
    """
    Adapt an opaque node to provide a consistent interface.
    """

    is_node_wrapper = True
    is_constant_scalar = False
    is_scalar = False

    child_attrs = []

    def __init__(self, pos, type, opaque_node, specialize_node_callback,
                 **kwds):
        super(NodeWrapper, self).__init__(pos, type)
        self.opaque_node = opaque_node
        self.specialize_node_callback = specialize_node_callback
        vars(self).update(kwds)

    def __hash__(self):
        return hash(self.opaque_node)

    def __eq__(self, other):
        if getattr(other, 'is_node_wrapper ', False):
            return self.opaque_node == other.opaque_node

        return NotImplemented

    def __deepcopy__(self, memo):
        kwds = dict(vars(self))
        kwds.pop('opaque_node')
        kwds = copy.deepcopy(kwds, memo)
        opaque_node = self.specialize_node_callback(self, memo)
        return type(self)(opaque_node=opaque_node, **kwds)

class BinaryOperationNode(ExprNode):
    child_attrs = ['lhs', 'rhs']
    def __init__(self, pos, type, lhs, rhs):
        super(BinaryOperationNode, self).__init__(pos, type)
        self.lhs, self.rhs = lhs, rhs

class BinopNode(BinaryOperationNode):

    is_binop = True

    def __init__(self, pos, type, operator, lhs, rhs):
        super(BinopNode, self).__init__(pos, type, lhs, rhs)
        self.operator = operator

    def is_cf_contig(self):
        c1, f1 = self.op1.is_cf_contig()
        c2, f2 = self.op2.is_cf_contig()
        return c1 and c2, f1 and f2

    @property
    def comparison_objects(self):
        return (self.operator, self.lhs, self.rhs)

class SingleOperandNode(ExprNode):
    child_attrs = ['operand']
    def __init__(self, pos, type, operand):
        super(SingleOperandNode, self).__init__(pos, type)
        self.operand = operand

class AssignmentExpr(BinaryOperationNode):
    is_assignment = True

class UnopNode(SingleOperandNode):

    is_unop = True

    def __init__(self, pos, type, operator, operand):
        super(UnopNode, self).__init__(pos, type, operand)
        self.operator = operator

    @property
    def comparison_objects(self):
        return (self.operator, self.operand)

class CastNode(SingleOperandNode):
    is_cast = True

class DereferenceNode(SingleOperandNode):
    is_dereference = True

class SingleIndexNode(BinaryOperationNode):
    is_index = True

class ConstantNode(ExprNode):
    is_constant = True
    def __init__(self, pos, type, value):
        super(ConstantNode, self).__init__(pos, type)
        self.value = value

class Variable(ExprNode):
    is_variable = True

    mangled_name = None

    def __init__(self, pos, type, name):
        super(Variable, self).__init__(pos, type)
        self.name = name

class ArrayAttribute(Variable):
    def __init__(self, pos, type, arrayvar):
        super(ArrayAttribute, self).__init__(pos, type,
                                             arrayvar.name + self._name)
        self.arrayvar = arrayvar

class DataPointer(ArrayAttribute):
    "Reference to the start of an array operand"
    _name = '_data'

class StridePointer(ArrayAttribute):
    "Reference to the stride pointer of an array variable operand"
    _name = '_strides'

class ShapePointer(ArrayAttribute):
    "Reference to the shape pointer of an array operand"
    _name = '_shape'

class TempNode(Variable):
    is_temp = True

class ErrorHandler(Node):
    """
    A node to handle errors. If there is an error handler in the outer scope,
    the specializer will first make this error handler generate disposal code
    for the wrapped AST body, and then jump to the error label of the parent
    error handler. At the outermost (function) level, the error handler simply
    returns an error indication.

        error_label: point to jump to in case of an error
        cleanup_label: point to jump to in the normal case

    It generates the following:

        error_var = 0;
        ...
        goto cleanup;
      error:
        error_var = 1;
      cleanup:
        ...
        if (error_var)
            goto outer_error_label;
    """
    child_attrs = ['error_var_init', 'body', 'cleanup_jump',
                   'error_target_label', 'error_set', 'cleanup_target_label',
                   'cascade']

    error_var_init = None
    cleanup_jump = None
    error_target_label = None
    error_set = None
    cleanup_target_label = None
    cascade = None

class JumpNode(Node):
    child_attrs = ['label']
    def __init__(self, pos, label):
        Node.__init__(self, pos)
        self.label = label

class JumpTargetNode(JumpNode):
    "A point to jump to"

class LabelNode(ExprNode):
    "A goto label or memory address that we can jump to"

    def __init__(self, pos, name):
        super(LabelNode, self).__init__(pos, None)
        self.name = name
        self.mangled_name = None
