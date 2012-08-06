try:
    from lxml import etree
    have_lxml = True
except ImportError:
    have_lxml = False
    try:
        # Python 2.5
        from xml.etree import cElementTree as etree
    except ImportError:
        try:
            # Python 2.5
            from xml.etree import ElementTree as etree
        except ImportError:
            try:
                # normal cElementTree install
                import cElementTree as etree
            except ImportError:
                # normal ElementTree install
                import elementtree.ElementTree as etree

import minivisitor

class XMLDumper(minivisitor.PrintTree):
    def visit_FunctionNode(self, node):
        self.treebuilder = etree.TreeBuilder()
        self.visit_Node(node)
        return self.treebuilder.close()

    def visit_Node(self, node):
        name = type(node).__name__
        format_value = self.format_value(node)
        if format_value:
            attrs = {
                'value': str(format_value),
                'id': hex(id(node)),
                'type': str(node.type)
            }
        else:
            attrs = None

        self.treebuilder.start(name, attrs)
        self.visitchildren(node)
        self.treebuilder.end(name)

def tostring(xml_root_element):
    et = etree.ElementTree(xml_root_element)
    kw = {}
    if have_lxml:
        kw['pretty_print'] = True

    return etree.tostring(et, encoding='UTF-8', **kw)