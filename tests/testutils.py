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
from ctypes_conversion import get_data_pointer, convert_to_ctypes

def getcontext():
    return miniast.CContext()

def get_llvm_context():
    return miniast.LLVMContext()

def specialize(specializer_cls, ast, context=None, print_tree=False):
    context = context or getcontext()
    specializers = [specializer_cls]
    result = iter(context.run(ast, specializers, print_tree=print_tree)).next()
    _, specialized_ast, _, code_result = result
    if not context.use_llvm:
        prototype, code_result = code_result
    return specialized_ast, code_result

def run(specializers, ast):
    context = getcontext()
    for result in context.run(ast, specializers):
        _, specialized_ast, _, (proto, impl) = result
        yield specialized_ast, impl

def build_vars(*types):
    return [b.variable(type, 'op%d' % i) for i, type in enumerate(types)]

def build_function(variables, body, name=None):
    return context.astbuilder.build_function(variables, body, name)

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
