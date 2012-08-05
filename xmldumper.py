import minivisitor

class XMLDumper(minivisitor.PrintTree):
    def visit_FunctionNode(self, node):
        self.treebuilder = etree.TreeBuidler()
        self.visit_Node(node)
        return self.treebuilder.close()

    def visit_Node(self, node):
        name = type(node).__name__
        format_value = self.format_value(node)
        if format_value:
            attrs = {'value': format_value}
        else:
            attrs = None

        self.treebuilder.start(name, attr)
        self.visitchildren(node)
        self.treebuilder.end(name)
