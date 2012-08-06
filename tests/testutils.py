"""
Module providing some test utilities.
"""

import os
import sys

import miniast
import specializers
import minitypes
import codegen
import xmldumper
import treepath

from minitypes import *
from xmldumper import etree, tostring

def getcontext():
    return miniast.CContext()

def specialize(specializer_cls, ast):
    context = getcontext()
    specializers = [specializer_cls]
    result = iter(context.run(ast, specializers)).next()
    _, specialized_ast, _, (proto, impl) = result
    return specialized_ast, impl

def run(specializers, ast):
    context = getcontext()
    for result in context.run(ast, specializers):
        _, specialized_ast, _, (proto, impl) = result
        yield specialized_ast, impl

def build_vars(*types):
    return [b.variable(type, 'op%d' % i) for i, type in enumerate(types)]

def build_function(variables, body, name=None):
    args = []
    for var in variables:
        if var.type.is_array:
            args.append(b.array_funcarg(var))
        else:
            args.append(b.funcarg(var))

    name = name or 'function'
    return b.function(name, body, args)

def toxml(function):
    return xmldumper.XMLDumper(context).visit(function)

def xpath(ast, expr):
    return treepath.find_all(ast, expr)

# Convenience variables
context = getcontext()
b = context.astbuilder

cinner = specializers.StridedCInnerContigSpecializer
cinner_sse = cinner.vectorized_equivalents[0]
ctiled = specializers.CTiledStridedSpecializer
contig = specializers.ContigSpecializer
contig_sse = contig.vectorized_equivalents[0]
