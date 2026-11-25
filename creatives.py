"""Simulated advertising creatives and platform-style ad previews.

The synthetic dataset contains performance numbers but no creative assets, so
this module generates *illustrative* creatives (headline, description, call to
action and artwork) deterministically from the campaign definition. Every
rendered preview carries a visible "simulated" label: these are not real
advertisements and were never published on any platform.
"""

from __future__ import annotations

import base64
import html
import zlib
from dataclasses import dataclass

import numpy as np

BRAND = "Lumen Goods"
BRAND_DOMAIN = "lumengoods.example"
SIMULATED_LABEL = "SIMULATED PREVIEW - NOT A REAL ADVERTISEMENT"

OBJECTIVE_PALETTES = {
    "Brand Awareness": ("#4F46E5", "#7C3AED"),
    "Conversions": ("#047857", "#10B981"),
    "Retargeting": ("#DC2626", "#F97316"),
    "Lead Generation": ("#0369A1", "#38BDF8"),
    "Product Launch": ("#BE185D", "#F59E0B"),
}
DEFAULT_PALETTE = ("#374151", "#9CA3AF")

HEADLINES = {
    "Brand Awareness": ["Everyday goods, thoughtfully made", "Meet the brand behind your next favourite thing",
                        "Quality you can feel from day one"],
    "Conversions": ["Free delivery on your first order", "Upgrade your routine - shop the bestsellers",
                    "Limited offer: save on top picks"],
    "Retargeting": ["Still thinking it over? Your cart is waiting", "Come back and complete your order",
                    "The items you liked are almost gone"],
    "Lead Generation": ["Get the buyer's guide - free", "Join the list for early access and offers",
                        "Request your personal product recommendation"],
    "Product Launch": ["Just launched: the new collection", "Be among the first to try it",
                       "Introducing our newest release"],
}
DESCRIPTIONS = {
    "Brand Awareness": "Discover what makes our products different.",
    "Conversions": "Fast checkout, easy returns and secure payment.",
    "Retargeting": "Pick up where you left off - your selections are saved.",
    "Lead Generation": "Takes under a minute. Unsubscribe any time.",
    "Product Launch": "New arrivals are in. Explore the range today.",
}
AUDIENCE_TAGLINES = {
    "New Customers": "New here? Start with a welcome offer.",
    "Returning Customers": "Welcome back - something new for you.",
    "Lookalike": "Loved by people with tastes like yours.",
    "Interest Based": "Picked for what you care about.",
}
CTAS = {
    "Brand Awareness": "Learn More", "Conversions": "Shop Now", "Retargeting": "Complete Order",
    "Lead Generation": "Sign Up", "Product Launch": "Discover Now",
}
ARTWORK_SIZES = {
    "Facebook": (600, 314), "Instagram": (400, 400), "TikTok": (270, 480),
    "YouTube": (480, 270), "Google": (120, 120),
}


@dataclass(frozen=True)
class Creative:
    campaign_id: str
    platform: str
    objective: str
    audience: str
    headline: str
    description: str
    cta: str
    primary: str
    secondary: str
    brand: str = BRAND
    is_simulated: bool = True


def _seed(campaign_id: str) -> int:
    return zlib.crc32(campaign_id.encode("utf-8"))


def build_creative(campaign_id: str, platform: str, objective: str, audience: str) -> Creative:
    """Deterministically generate a simulated creative for a campaign."""
    rng = np.random.default_rng(_seed(campaign_id))
    pool = HEADLINES.get(objective, ["Discover something new"])
    base = DESCRIPTIONS.get(objective, "Learn more about our products.")
    tagline = AUDIENCE_TAGLINES.get(audience)
    primary, secondary = OBJECTIVE_PALETTES.get(objective, DEFAULT_PALETTE)
    return Creative(
        campaign_id=campaign_id, platform=platform, objective=objective, audience=audience,
        headline=pool[int(rng.integers(0, len(pool)))],
        description=f"{base} {tagline}" if tagline else base,
        cta=CTAS.get(objective, "Learn More"),
        primary=primary, secondary=secondary,
    )


def render_artwork_svg(creative: Creative, width: int, height: int) -> str:
    """Abstract product-style artwork generated from the campaign id."""
    rng = np.random.default_rng(_seed(creative.campaign_id) + 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{creative.primary}"/>'
        f'<stop offset="1" stop-color="{creative.secondary}"/></linearGradient></defs>',
        f'<rect width="{width}" height="{height}" fill="url(#g)"/>',
    ]
    for _ in range(7):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        radius = rng.uniform(0.08, 0.30) * min(width, height)
        opacity = rng.uniform(0.10, 0.28)
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="#FFFFFF" opacity="{opacity:.2f}"/>')
    side = 0.42 * min(width, height)
    x, y = (width - side) / 2, (height - side) / 2
    parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{side:.1f}" height="{side:.1f}" rx="{side * 0.22:.1f}" '
                 f'fill="#FFFFFF" opacity="0.92"/>')
    parts.append(f'<circle cx="{width / 2:.1f}" cy="{height / 2:.1f}" r="{side * 0.26:.1f}" fill="{creative.primary}" opacity="0.9"/>')
    parts.append("</svg>")
    return "".join(parts)


def _data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def _esc(value: str) -> str:
    return html.escape(str(value), quote=True)


def _avatar(creative: Creative, size: int = 36) -> str:
    return (f'<div style="width:{size}px;height:{size}px;border-radius:50%;background:{_esc(creative.primary)};'
            f'color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;">'
            f'{_esc(creative.brand[0])}</div>')


def _button(creative: Creative, background: str, radius: int = 6) -> str:
    return (f'<span style="display:inline-block;padding:7px 14px;border-radius:{radius}px;background:{background};'
            f'color:#fff;font-weight:600;font-size:13px;">{_esc(creative.cta)}</span>')


def _image(creative: Creative, width: int, height: int, extra_style: str = "") -> str:
    uri = _data_uri(render_artwork_svg(creative, width, height))
    return (f'<img alt="Simulated creative artwork" src="{uri}" '
            f'style="width:100%;display:block;aspect-ratio:{width}/{height};object-fit:cover;{extra_style}"/>')


def _feed_card(creative: Creative, width: int, height: int, meta: str, footer_cta_bg: str) -> str:
    return (
        f'{_header(creative, meta)}'
        f'<div style="padding:0 12px 10px;font-size:14px;color:#1f2937;">{_esc(creative.description)}</div>'
        f'{_image(creative, width, height)}'
        f'<div style="display:flex;justify-content:space-between;align-items:center;padding:10px 12px;background:#f3f4f6;">'
        f'<div><div style="font-size:11px;color:#6b7280;text-transform:uppercase;">{_esc(BRAND_DOMAIN)}</div>'
        f'<div style="font-weight:700;font-size:14px;color:#111827;">{_esc(creative.headline)}</div></div>'
        f'{_button(creative, footer_cta_bg)}</div>'
    )


def _header(creative: Creative, meta: str) -> str:
    return (f'<div style="display:flex;gap:10px;align-items:center;padding:10px 12px;">{_avatar(creative)}'
            f'<div><div style="font-weight:700;font-size:14px;color:#111827;">{_esc(creative.brand)}</div>'
            f'<div style="font-size:12px;color:#6b7280;">{_esc(meta)}</div></div></div>')


def _instagram_card(creative: Creative) -> str:
    width, height = ARTWORK_SIZES["Instagram"]
    return (
        f'{_header(creative, "Sponsored")}{_image(creative, width, height)}'
        f'<div style="padding:10px 12px;background:#0095f6;color:#fff;font-weight:600;font-size:14px;">'
        f'{_esc(creative.cta)} &rsaquo;</div>'
        f'<div style="padding:10px 12px;font-size:14px;color:#111827;"><b>{_esc(creative.brand)}</b> '
        f'{_esc(creative.headline)}. {_esc(creative.description)}</div>'
    )


def _tiktok_card(creative: Creative) -> str:
    width, height = ARTWORK_SIZES["TikTok"]
    return (
        f'<div style="position:relative;background:#000;">{_image(creative, width, height)}'
        f'<div style="position:absolute;left:0;right:0;bottom:0;padding:14px;color:#fff;'
        f'background:linear-gradient(transparent,rgba(0,0,0,0.75));">'
        f'<div style="font-weight:700;">@{_esc(creative.brand.replace(" ", "").lower())}</div>'
        f'<div style="font-size:13px;margin:4px 0 8px;">{_esc(creative.headline)}</div>'
        f'<div style="font-size:11px;opacity:0.85;margin-bottom:8px;">Sponsored</div>'
        f'{_button(creative, "#fe2c55", 4)}</div></div>'
    )


def _youtube_card(creative: Creative) -> str:
    width, height = ARTWORK_SIZES["YouTube"]
    return (
        f'<div style="position:relative;">{_image(creative, width, height)}'
        f'<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;">'
        f'<div style="width:56px;height:40px;border-radius:10px;background:rgba(255,0,0,0.9);display:flex;'
        f'align-items:center;justify-content:center;color:#fff;font-size:20px;">&#9654;</div></div>'
        f'<div style="position:absolute;left:8px;bottom:8px;background:rgba(0,0,0,0.75);color:#fff;'
        f'font-size:11px;padding:2px 6px;border-radius:3px;">Ad &middot; 0:15</div></div>'
        f'<div style="padding:12px;display:flex;justify-content:space-between;align-items:center;gap:8px;">'
        f'<div><div style="font-weight:700;font-size:14px;color:#111827;">{_esc(creative.headline)}</div>'
        f'<div style="font-size:12px;color:#6b7280;">{_esc(creative.brand)} &middot; Sponsored</div></div>'
        f'{_button(creative, "#065fd4", 18)}</div>'
    )


def _google_card(creative: Creative) -> str:
    return (
        f'<div style="padding:16px;">'
        f'<div style="font-size:13px;color:#202124;"><b>Sponsored</b> &middot; {_esc(BRAND_DOMAIN)}</div>'
        f'<div style="font-size:20px;color:#1a0dab;margin:4px 0;">{_esc(creative.headline)} | {_esc(creative.brand)}</div>'
        f'<div style="font-size:14px;color:#4d5156;">{_esc(creative.description)}</div>'
        f'<div style="margin-top:10px;">{_button(creative, "#1a73e8", 4)}</div></div>'
    )


_RENDERERS = {
    "Facebook": lambda c: _feed_card(c, *ARTWORK_SIZES["Facebook"], "Sponsored", "#1877f2"),
    "Instagram": _instagram_card,
    "TikTok": _tiktok_card,
    "YouTube": _youtube_card,
    "Google": _google_card,
}
_MAX_WIDTH = {"Facebook": 420, "Instagram": 360, "TikTok": 260, "YouTube": 420, "Google": 480}


def render_ad_preview_html(creative: Creative) -> str:
    """Self-contained HTML for a platform-style preview, labelled as simulated."""
    body = _RENDERERS.get(creative.platform, lambda c: _feed_card(c, 600, 314, "Sponsored", "#374151"))(creative)
    width = _MAX_WIDTH.get(creative.platform, 420)
    return (
        f'<div style="max-width:{width}px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">'
        f'<div style="border:1px solid #d1d5db;border-radius:10px;overflow:hidden;background:#fff;">{body}</div>'
        f'<div style="margin-top:6px;font-size:11px;letter-spacing:0.04em;color:#b45309;font-weight:600;">'
        f'{_esc(SIMULATED_LABEL)}</div></div>'
    )
