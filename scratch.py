from model.graph import Edge, Node

edge1 = Edge((Node(0), Node(1)))
edge2 = Edge((Node(1), Node(0)))

print(edge1)
print(edge2)
print(edge1 == edge2)
