from __future__ import annotations

import ast
import operator

# A deliberately tiny, safe arithmetic evaluator. We do NOT use eval() — only an
# allow-list of arithmetic AST node types, so the agent can compute ratios and
# growth percentages without any risk of executing arbitrary code.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_calc(expression: str) -> float:
    """Evaluate an arithmetic expression. Raises ValueError on anything unsafe."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    tree = ast.parse(expression, mode="eval")
    return _eval(tree.body)
