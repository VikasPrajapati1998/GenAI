"""
mcp_procurement_server.py — an MCP server exposing procurement tools and a policy resource
=========================================================================================

Used by MCPAgent.py (which launches this file as a stdio subprocess). You can also inspect it with
the MCP Inspector:   pip install "mcp[cli]"   then   mcp dev mcp_procurement_server.py

Written for the MCP Python SDK v2 (`from mcp.server import MCPServer`), which implements the stateless
2026-07-28 spec. With SDK 1.x the equivalent import is `from mcp.server.fastmcp import FastMCP`.

Tools          get_quotes(product) · check_vendor(vendor) · approval_level(amount_inr) · raise_purchase_request(...)
Resource       policy://procurement      (full policy text, readable by the host)
"""

from mcp.server import MCPServer

mcp = MCPServer("procurement")

VENDORS = {
    "ABC Technologies": {"status": "APPROVED", "iso_valid_until": "2027-03-31", "parent": "Helix Holdings (under investigation)"},
    "Zenith Systems": {"status": "APPROVED", "iso_valid_until": "2026-10-31", "parent": None},
    "Nova Compute": {"status": "BLACKLISTED", "iso_valid_until": None, "parent": "Helix Holdings (under investigation)"},
    "Orbit Infotech": {"status": "APPROVED", "iso_valid_until": "2028-01-15", "parent": None},
}
QUOTES = {
    "RTX 6000 Ada": {"ABC Technologies": 650000, "Zenith Systems": 620000, "Nova Compute": 590000, "Orbit Infotech": 668000},
    "RTX PRO 6000 Blackwell": {"Orbit Infotech": 910000},
}
POLICY = """Procurement Policy 3.2
§1 Up to INR 2,00,000: department head. INR 2,00,001–10,00,000: Finance Controller. Above INR 10,00,000: CFO + 3 quotes.
§4 Only APPROVED vendors with an unexpired ISO 9001 certificate may receive hardware POs.
§5 Blacklisted vendors are barred for 24 months; vendors whose parent is under investigation need Legal clearance."""
TODAY = "2026-11-20"


@mcp.tool()
def get_quotes(product: str) -> dict:
    """Return current vendor quotes (INR per unit) for a product, e.g. 'RTX 6000 Ada'."""
    for name, quotes in QUOTES.items():
        if product.lower() in name.lower() or name.lower() in product.lower():
            return {"product": name, "quotes_inr": quotes}
    return {"error": f"no quotes for '{product}'", "known_products": list(QUOTES)}


@mcp.tool()
def check_vendor(vendor: str) -> dict:
    """Check a vendor's eligibility against policy §4/§5. Returns status, certificate validity and issues."""
    for name, v in VENDORS.items():
        if vendor.lower() in name.lower():
            issues = []
            if v["status"] != "APPROVED":
                issues.append(f"status {v['status']}")
            if not v["iso_valid_until"] or v["iso_valid_until"] < TODAY:
                issues.append(f"ISO 9001 not valid on {TODAY}")
            if v["parent"] and "investigation" in v["parent"]:
                issues.append("parent under investigation: Legal clearance required")
            return {"vendor": name, **v, "eligible": not issues, "issues": issues}
    return {"error": f"unknown vendor '{vendor}'"}


@mcp.tool()
def approval_level(amount_inr: float) -> dict:
    """Return who must approve a purchase of the given total amount in INR (policy §1)."""
    a = float(amount_inr)
    who = "Department head" if a <= 200000 else "Finance Controller" if a <= 1000000 else "CFO + three quotes"
    return {"amount_inr": a, "approver": who}


@mcp.tool()
def raise_purchase_request(vendor: str, product: str, quantity: int, confirmed_by_human: bool = False) -> dict:
    """Create a purchase request. A WRITE action: refuses unless confirmed_by_human is true."""
    if not confirmed_by_human:
        return {"status": "BLOCKED", "reason": "human confirmation required before creating a purchase request"}
    return {"status": "CREATED", "request_id": "PR-2026-1187", "vendor": vendor, "product": product, "quantity": quantity}


@mcp.resource("policy://procurement")
def procurement_policy() -> str:
    """The full procurement policy text."""
    return POLICY


if __name__ == "__main__":
    mcp.run()          # stdio transport by default; use mcp.run("streamable-http") to serve over HTTP
