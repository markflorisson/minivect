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

class UndocClassAttribute(object):
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return self.cls(*args, **kwargs)

class Context(object):
    """
    A context that knows how to map ASTs back and forth, how to wrap nodes
    and types, and how to instantiate a code generator for specialization.

    An opaque_node or foreign node is a node that is not from our AST,
    and a normal node is one that has a interface compatible with ours.

    To provide custom functionality, set the following attributes, or
    subclass this class.

    :param astbuilder: the :py:class:`ASTBuilder` or ``None``
    :param typemapper: the :py:class:`minivect.minitypes.Typemapper` or
                       ``None`` for the default.

    .. attribute:: codegen_cls

        The code generator class that is used to generate code.
        The default is :py:class:`minivect.codegen.CodeGen`

    .. attribute:: cleanup_codegen_cls

        The code generator that generates code to dispose of any
        garbage (e.g. intermediate object temporaries).
        The default is :py:class:`minivect.codegen.CodeGenCleanup`

    .. attribute:: codewriter_cls

        The code writer that the code generator writes its generated code
        to. This may be strings or arbitrary objects.
        The default is :py:class:`minivect.minicode.CodeWriter`, which accepts
        arbitrary objects.

    .. attribute:: codeformatter_cls

        A formatter to format the generated code.

        The default is :py:class:`minivect.minicode.CodeFormatter`,
        which returns a list of objects written. Set this to
        :py:class:`minivect.minicode.CodeStringFormatter`
        to have the strings joined together.

    .. attribute:: specializer_mixin_cls

        A specializer mixin class that can override or intercept
        functionality. This class should likely participate
        cooperatively in MI.

    Use subclass :py:class:`CContext` to get the defaults for C code generation.
    """

    debug = False

    codegen_cls = UndocClassAttribute(codegen.CodeGen)
    cleanup_codegen_cls = UndocClassAttribute(codegen.CodeGenCleanup)
    codewriter_cls = UndocClassAttribute(minicode.CodeWriter)
    codeformatter_cls = UndocClassAttribute(minicode.CodeFormatter)

    specializer_mixin_cls = None

    def __init__(self, astbuilder=None, typemapper=None):
        self.astbuilder = astbuilder or ASTBuilder(self)
        self.typemapper = typemapper or minitypes.TypeMapper(self)

    def run_opaque(self, astmapper, opaque_ast, specializers):
        return self.run(astmapper.visit(opaque_ast), specializers)

    def run(self, ast, specializer_classes):
        """
        Specialize the given AST with all given specializers and return
        an iterable of generated code in the form of
        ``(specializer, new_ast, codewriter, code_obj)``

        The code_obj is the generated code (e.g. a string of C code),
        depending on the code formatter used.
        """
        for specializer_class in specializer_classes:
            if self.specializer_mixin_cls:
                cls1, cls2 = self.specializer_mixin_cls, specializer_class
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
        "Run the disposal code generator on an (sub)AST"
        transform = self.cleanup_codegen_cls(self, code)
        transform.visit(node)

    #
    ### Override in subclasses where needed
    #

    def promote_types(self, type1, type2):
        "Promote types in an arithmetic operation"
        return self.typemapper.promote_types(type1, type2)

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
        raise NotImplementedError

    def to_llvm(self, type):
        "Return an LLVM type for the given minitype"
        return self.typemapper.to_llvm(type)


class CContext(Context):
    "Set defaults for C code generation."

    codegen_cls = codegen.CCodeGen
    codewriter_cls = minicode.CCodeWriter
    codeformatter_cls = minicode.CCodeStringFormatter

class ASTBuilder(object):
    """
    This class is used to build up a minivect AST. It can be used by a user
    from a transform or otherwise, but the important bit is that we use it
    in our code to build up an AST that can be overridden by the user,
    and which makes it convenient to build up complex ASTs concisely.
    """

    # the 'pos' attribute is set for each visit to each node by
    # the ASTMapper
    pos = None

    def __init__(self, context):
        """
        :param context: the :py:class:`Context`
        """
        self.context = context

    def _infer_type(self, value):
        "Used to infer types for self.constant()"
        if isinstance(value, (int, long)):
            return minitypes.IntType()
        elif isinstance(value, float):
            return minitypes.FloatType()
        elif isinstance(value, str):
            return minitypes.CStringType()
        else:
            raise minierror.InferTypeError()

    def function(self, name, body, args, shapevar=None, posinfo=None):
        """
        Create a new function.

        :type name: str
        :param name: name of the function

        :type args: [:py:class:`FunctionArgument`]
        :param args: all array and scalar arguments to the function, excluding
                     shape or position information.

        :param shapevar: the :py:class:`Variable` for the total broadcast shape
                         If ``None``, a default of ``Py_ssize_t *`` is assumed.

        :type posinfo: :py:class:`FunctionArgument`
        :param posinfo: if given, this will be the second, third and fourth
                        arguments to the function ``(filename, lineno, column)``.
        """
        if shapevar is None:
            shapevar = self.variable(minitypes.Py_ssize_t.pointer(),
                                     '__pyx_shape')

        arguments, scalar_arguments = [], []
        for arg in args:
            if arg.type.is_array:
                arguments.append(arg)
            else:
                scalar_arguments.append(arg)

        arguments.insert(0, self.funcarg(shapevar))
        if posinfo:
            arguments.insert(1, posinfo)
        body = self.nditerate(body)
        return FunctionNode(self.pos, name, body, arguments, scalar_arguments,
                            shapevar, posinfo,
                            error_value=self.constant(-1),
                            success_value=self.constant(0))

    def funcarg(self, variable, *variables, **kwargs):
        """
        Create a (compound) function argument consisting of one or multiple
        argument Variables.
        """
        if not variables:
            variables = [variable]
        return FunctionArgument(self.pos, variable, list(variables))

    def array_funcarg(self, variable):
        "Create an array function argument"
        return ArrayFunctionArgument(
                self.pos, variable.type, name=variable.name,
                variable=variable,
                data_pointer=self.data_pointer(variable),
                #shape_pointer=self.shapevar(variable),
                strides_pointer=self.stridesvar(variable))

    def incref(self, var, funcname='Py_INCREF'):
        "Generate a Py_INCREF() statement"
        functype = minitypes.FunctionType(return_type=minitypes.void,
                                          args=[minitypes.object_])
        py_incref = self.funcname(functype, funcname)
        return self.expr_stat(self.funccall(py_incref, [var]))

    def decref(self, var):
        "Generate a Py_DECCREF() statement"
        return self.incref(var, funcname='Py_DECREF')

    def print_(self, *args):
        "Print out all arguments to stdout"
        return PrintNode(self.pos, args=list(args))

    def funccall(self, func_or_pointer, args):
        """
        Generate a call to the given function (a :py:class:`FuncNameNode`) of
        :py:class:`minivect.minitypes.FunctionType` or a
        pointer to a function type and the given arguments.
        """
        type = func_or_pointer.type
        if type.is_pointer:
            type = func_or_pointer.type.base_type
        return FuncCallNode(self.pos, type,
                            func_or_pointer=func_or_pointer, args=args)

    def funcname(self, type, name):
        return FuncNameNode(self.pos, type, name=name)

    def nditerate(self, body):
        """
        This node wraps the given AST expression in an :py:class:`NDIterate`
        node, which will be expanded by the specializers to one or several
        loops.
        """
        return NDIterate(self.pos, body)

    def for_(self, body, init, condition, step, is_tiled=False):
        """
        Create a for loop node.

        :param body: loop body
        :param init: assignment expression
        :param condition: boolean loop condition
        :param step: step clause (assignment expression)
        """
        return ForNode(self.pos, init, condition, step, body, is_tiled)

    def for_range_upwards(self, body, upper, lower=None, step=None):
        """
        Create a single upwards for loop, typically used from a specializer to
        replace an :py:class:`NDIterate` node.

        :param body: the loop body
        :param upper: expression specifying an upper bound
        """
        if lower is None:
            lower = self.constant(0)
        if step is None:
            step = self.constant(1)

        temp = self.temp(minitypes.Py_ssize_t)
        init = self.assign_expr(temp, lower)
        condition = self.binop(minitypes.bool_, '<', temp, upper)
        step = self.assign_expr(temp, self.add(temp, step))

        result = self.for_(body, init, condition, step)
        result.target = temp
        return result

    def omp_for(self, for_node, if_clause):
        """
        Annotate the for loop with an OpenMP parallel for clause.

        :param if_clause: the expression node that determines whether the
                          parallel section is executed or whether it is
                          executed sequentially (to avoid synchronization
                          overhead)
        """
        return OpenMPLoopNode(self.pos, for_node=for_node,
                              if_clause=if_clause)

    def pragma_for(self, for_node):
        """
        Annotate the for loop with pragmas.
        """
        return PragmaForLoopNode(self.pos, for_node=for_node)

    def stats(self, *statements):
        """
        Wrap a bunch of statements in an AST node.
        """
        stats = []
        for stat in statements:
            if stat.is_statlist:
                stats.extend(stat.stats)
            else:
                stats.append(stat)

        return StatListNode(self.pos, stats)

    def expr_stat(self, expr):
        "Turn an expression into a statement"
        return ExprStatNode(expr.pos, type=expr.type, expr=expr)

    def expr(self, stats=(), expr=None):
        "Evaluate a bunch of statements before evaluating an expression."
        return ExprNodeWithStatement(self.pos, type=expr.type,
                                     stat=self.stats(*stats), expr=expr)

    def if_(self, cond, body):
        "If statement"
        return IfNode(self.pos, cond=cond, body=body)

    def if_else_expr(self, cond, lhs, rhs):
        "If/else expression, resulting in lhs if cond else rhs"
        type = self.context.promote_types(lhs.type, rhs.type)
        return IfElseExprNode(self.pos, type=type, cond=cond, lhs=lhs, rhs=rhs)

    def binop(self, type, op, lhs, rhs):
        """
        Binary operation on two nodes.

        :param type: the result type of the expression
        :param op: binary operator
        :type op: str
        """
        return BinopNode(self.pos, type, op, lhs, rhs)

    def add(self, lhs, rhs, result_type=None):
        """
        Shorthand for the + binop. Filters out adding 0 constants.
        """
        if lhs.is_constant and lhs.value == 0:
            return rhs
        elif rhs.is_constant and rhs.value == 0:
            return lhs

        if result_type is None:
            result_type = self.context.promote_types(lhs.type, rhs.type)
        return self.binop(result_type, '+', lhs, rhs)

    def mul(self, lhs, rhs, result_type=None, op='*'):
        """
        Shorthand for the * binop. Filters out multiplication with 1 constants.
        """
        if op == '*' and lhs.is_constant and lhs.value == 1:
            return rhs
        elif rhs.is_constant and rhs.value == 1:
            return lhs

        if result_type is None:
            result_type = self.context.promote_types(lhs.type, rhs.type)
        return self.binop(result_type, op, lhs, rhs)

    def div(self, lhs, rhs, result_type=None):
        return self.mul(lhs, rhs, result_type=result_type, op='/')

    def min(self, lhs, rhs):
        """
        Returns min(lhs, rhs) expression.

        .. NOTE:: Make lhs and rhs temporaries if they should only be
                  evaluated once.
        """
        type = self.context.promote_types(lhs.type, rhs.type)
        cmp_node = self.binop(type, '<', lhs, rhs)
        return self.if_else_expr(cmp_node, lhs, rhs)

    def index(self, pointer, index, dest_pointer_type=None):
        """
        Index a pointer with the given index node.

        :param dest_pointer_type: if given, cast the result (*after* adding
                                  the index) to the destination type and
                                  dereference.
        """
        if dest_pointer_type:
            return self.index_multiple(pointer, [index], dest_pointer_type)
        return SingleIndexNode(self.pos, pointer.type.base_type,
                               pointer, index)

    def index_multiple(self, pointer, indices, dest_pointer_type=None):
        """
        Same as :py:meth:`index`, but accepts multiple indices. This is
        useful e.g. after multiplication of the indices with the strides.
        """
        for index in indices:
            pointer = self.add(pointer, index)

        if dest_pointer_type is not None:
            pointer = self.cast(pointer, dest_pointer_type)

        return self.dereference(pointer)

    def assign_expr(self, node, value):
        "Create an assignment expression assigning ``value`` to ``node``"
        assert node is not None
        if not isinstance(value, Node):
            value = self.constant(value)
        return AssignmentExpr(self.pos, node.type, node, value)

    def assign(self, node, value):
        "Assignment statement"
        return self.expr_stat(self.assign_expr(node, value))

    def dereference(self, pointer):
        "Dereference a pointer"
        return DereferenceNode(self.pos, pointer.type.base_type, pointer)

    def unop(self, type, operator, operand):
        "Unary operation. ``type`` indicates the result type of the expression."
        return UnopNode(self.pos, type, operator, operand)

    def coerce_to_temp(self, expr):
        "Coerce the given expression to a temporary"
        type = expr.type
        if type.is_array:
            type = type.dtype
        temp = self.temp(type)
        return self.expr(stats=[self.assign(temp, expr)], expr=temp)

    def temp(self, type, name=None, rhs=None):
        "Allocate a temporary of a given type"
        return TempNode(self.pos, type, name=name or 'temp', rhs=rhs)

    def constant(self, value, type=None):
        """
        Create a constant from a Python value. If type is not given, it is
        inferred (or it will raise a
        :py:class:`minivect.minierror.InferTypeError`).
        """
        if type is None:
            type = self._infer_type(value)
        return ConstantNode(self.pos, type, value)

    def variable(self, type, name):
        """
        Create a variable with a name and type. Variables
        may refer to function arguments, functions, etc.
        """
        return Variable(self.pos, type, name)

    def cast(self, node, dest_type):
        "Cast node to the given destination type"
        return CastNode(self.pos, dest_type, node)

    def return_(self, result):
        "Return a result"
        return ReturnNode(self.pos, result)

    def data_pointer(self, variable):
        "Return the data pointer of an array variable"
        assert variable.type.is_array
        return DataPointer(self.pos, variable.type.dtype.pointer(),
                           variable)

    def shape_index(self, index, function):
        "Index the shape of the array operands with integer `index`"
        return self.index(function.shape, self.constant(index))

    def extent(self, variable, index, function):
        "Index the shape of a specific variable with integer `index`"
        assert variable.type.is_array
        offset = function.ndim - variable.type.ndim
        return self.index(function.shape, self.constant(index + offset))

    def stridesvar(self, variable):
        "Return the strides variable for the given array operand"
        return StridePointer(self.pos, minitypes.Py_ssize_t.pointer(), variable)

    def stride(self, variable, index):
        "Return the stride of array operand `variable` at integer `index`"
        return self.index(self.stridesvar(variable), self.constant(index))

    def sizeof(self, type):
        "Return the expression sizeof(type)"
        return SizeofNode(self.pos, minitypes.size_t, sizeof_type=type)

    def jump(self, label):
        "Jump to a label"
        return JumpNode(self.pos, label)

    def jump_target(self, label):
        """
        Return a target that can be jumped to given a label. The label is
        shared between the jumpers and the target.
        """
        return JumpTargetNode(self.pos, label)

    def label(self, name):
        "Return a label with a name"
        return LabelNode(self.pos, name)

    def raise_exc(self, posinfo, exc_var, msg_val, fmt_args):
        """
        Raise an exception given the positional information (see the `posinfo`
        method), the exception type (PyExc_*), a formatted message string and
        a list of values to be used for the format string.
        """
        return RaiseNode(self.pos, posinfo, exc_var, msg_val, fmt_args)

    def posinfo(self, posvars):
        """
        Return position information given a list of position variables
        (filename, lineno, column). This can be used for raising exceptions.
        """
        return PositionInfoNode(self.pos, posinfo=posvars)

    def error_handler(self, node):
        """
        Wrap the given node, which may raise exceptions, in an error handler.
        An error handler allows the code to clean up before propagating the
        error, and finally returning an error indicator from the function.
        """
        return ErrorHandler(self.pos, body=node,
                            error_label=self.label('error'),
                            cleanup_label=self.label('cleanup'))

    def wrap(self, opaque_node, specialize_node_callback, **kwds):
        """
        Wrap a node and type and return a NodeWrapper node. This node
        will have to be handled by the caller in a code generator. The
        specialize_node_callback is called when the NodeWrapper is
        specialized by a Specializer.
        """
        type = minitypes.TypeWrapper(self.context.gettype(opaque_node),
                                     self.context)
        return NodeWrapper(self.context.getpos(opaque_node), type,
                           opaque_node, specialize_node_callback, **kwds)

class Position(object):
    "Each node has a position which is an instance of this type."

    def __init__(self, filename, line, col):
        self.filename = filename
        self.line = line
        self.col = col

    def __str__(self):
        return "%s:%d:%d" % (self.filename, self.line, self.col)

class Node(miniutils.ComparableObjectMixin):
    """
    Base class for AST nodes.
    """

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
    is_temp = False
    is_statement = False

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
        for obj in self.comparison_objects:
            h = h ^ hash(subtype)

        return h

class ExprNode(Node):
    "Base class for expressions. Each node has a type."

    is_expression = True

    def __init__(self, pos, type, **kwds):
        super(ExprNode, self).__init__(pos, **kwds)
        self.type = type

class FunctionNode(Node):
    """
    Function node. error_value and success_value are returned in case of
    exceptions and success respectively.

    .. attribute:: shape
            the broadcast shape for all operands

    .. attribute:: ndim
            the ndim of the total broadcast' shape

    .. attribute:: arguments
            all array arguments

    .. attribute:: scalar arguments
        all non-array arguments

    .. attribute:: posinfo
        the position variables we can write to in case of an exception
    """

    child_attrs = ['body', 'arguments', 'scalar_arguments']

    def __init__(self, pos, name, body, arguments, scalar_arguments,
                 shape, posinfo, error_value, success_value):
        super(FunctionNode, self).__init__(pos)
        self.name = name
        self.body = body
        self.arguments = arguments
        self.scalar_arguments = scalar_arguments
        self.shape = shape
        self.posinfo = posinfo
        self.error_value = error_value
        self.success_value = success_value

        self.args = dict((v.name, v) for v in arguments)
        self.ndim = max(arg.type.ndim for arg in arguments
                                          if arg.type and arg.type.is_array)


class FuncCallNode(ExprNode):
    """
    Call a function given a pointer or its name (FuncNameNode)
    """

    child_attrs = ['func_or_pointer', 'args']

class FuncNameNode(ExprNode):
    """
    Load an external function by its name.
    """
    name = None

class ReturnNode(Node):
    "Return an operand"

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

    .. attribute:: variable

        some argument to the function (array or otherwise)

    .. attribute:: variables

        the actual variables this operand should be unpacked into
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
    "Array operand to the function"

    child_attrs = ['data_pointer', 'strides_pointer']
    is_array_funcarg = True

class PrintNode(Node):
    "Print node for some arguments"

    child_attrs = ['args']

class NDIterate(Node):
    """
    Iterate in N dimensions. See :py:class:`ASTBuilder.nditerate`
    """

    child_attrs = ['body']

    def __init__(self, pos, body):
        super(NDIterate, self).__init__(pos)
        self.body = body

class ForNode(Node):
    """
    A for loop, see :py:class:`ASTBuilder.for_`
    """

    child_attrs = ['init', 'condition', 'step', 'body']

    def __init__(self, pos, init, condition, step, body, is_tiled=False):
        super(ForNode, self).__init__(pos)
        self.init = init
        self.condition = condition
        self.step = step
        self.body = body
        self.is_tiled = False

class IfNode(Node):
    "An 'if' statement, see A for loop, see :py:class:`ASTBuilder.if_"

    child_attrs = ['cond', 'body']

class StatListNode(Node):
    """
    A node to wrap multiple statements, see :py:class:`ASTBuilder.stats
    """
    child_attrs = ['stats']
    is_statlist = True

    def __init__(self, pos, statements):
        super(StatListNode, self).__init__(pos)
        self.stats = statements

class ExprStatNode(Node):
    "Turn an expression into a statement, see :py:class:`ASTBuilder.expr_stat`"
    child_attrs = ['expr']
    is_statement = True

class ExprNodeWithStatement(Node):
    child_attrs = ['stat', 'expr']

class NodeWrapper(ExprNode):
    """
    Adapt an opaque node to provide a consistent interface. This has to be
    handled by the user's specializer. See :py:class:`ASTBuilder.wrap`
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
    "Base class for binary operations"
    child_attrs = ['lhs', 'rhs']
    def __init__(self, pos, type, lhs, rhs):
        super(BinaryOperationNode, self).__init__(pos, type)
        self.lhs, self.rhs = lhs, rhs

class BinopNode(BinaryOperationNode):
    "Node for binary operations"

    is_binop = True

    def __init__(self, pos, type, operator, lhs, rhs):
        super(BinopNode, self).__init__(pos, type, lhs, rhs)
        self.operator = operator

    @property
    def comparison_objects(self):
        return (self.operator, self.lhs, self.rhs)

class SingleOperandNode(ExprNode):
    "Base class for operations with one operand"
    child_attrs = ['operand']
    def __init__(self, pos, type, operand):
        super(SingleOperandNode, self).__init__(pos, type)
        self.operand = operand

class AssignmentExpr(BinaryOperationNode):
    is_assignment = True

class IfElseExprNode(ExprNode):
    child_attrs = ['cond', 'lhs', 'rhs']

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

class SizeofNode(ExprNode):
    is_sizeof = True

class Variable(ExprNode):
    """
    Represents use of a function argument in the function.
    """

    is_variable = True
    mangled_name = None

    def __init__(self, pos, type, name, **kwargs):
        super(Variable, self).__init__(pos, type, **kwargs)
        self.name = name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

class ArrayAttribute(Variable):
    "Denotes an attribute of array operands, e.g. the data or stride pointers"
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

#class ShapePointer(ArrayAttribute):
#    "Reference to the shape pointer of an array operand."
#    _name = '_shape'

class TempNode(Variable):
    "A temporary of a certain type"

    child_attrs = ['rhs']

    is_temp = True

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

class OpenMPLoopNode(Node):
    """
    Execute a loop in parallel.
    """
    child_attrs = ['for_node', 'if_clause']

class PragmaForLoopNode(Node):
    """
    Generate compiler-specific pragmas to aid things like SIMDization.
    """
    child_attrs = ['for_node']

class ErrorHandler(Node):
    """
    A node to handle errors. If there is an error handler in the outer scope,
    the specializer will first make this error handler generate disposal code
    for the wrapped AST body, and then jump to the error label of the parent
    error handler. At the outermost (function) level, the error handler simply
    returns an error indication.

    .. attribute:: error_label

        point to jump to in case of an error

    .. attribute:: cleanup_label

        point to jump to in the normal case

    It generates the following:

    .. code-block:: c

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
    "A jump to a jump target"
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
