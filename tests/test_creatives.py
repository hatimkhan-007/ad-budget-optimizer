import dataclasses
import xml.etree.ElementTree as ET

import pytest

from creatives import (ARTWORK_SIZES, SIMULATED_LABEL, build_creative, render_ad_preview_html, render_artwork_svg)

PLATFORMS = ["Facebook", "Instagram", "TikTok", "YouTube", "Google"]


def test_creative_is_deterministic_and_complete():
    first = build_creative("FB-CONV-NEW", "Facebook", "Conversions", "New Customers")
    second = build_creative("FB-CONV-NEW", "Facebook", "Conversions", "New Customers")
    assert first == second
    assert first.headline and first.description and first.cta == "Shop Now"
    assert first.is_simulated


def test_creative_handles_unknown_objective_and_audience():
    creative = build_creative("X-1", "Facebook", "Mystery", "Nobody")
    assert creative.headline and creative.cta == "Learn More"


def test_different_campaigns_can_have_different_creatives():
    headlines = {build_creative(f"FB-CONV-{i}", "Facebook", "Conversions", "Lookalike").headline for i in range(30)}
    assert len(headlines) > 1


@pytest.mark.parametrize("platform", PLATFORMS + ["Pinterest"])
def test_every_preview_is_labelled_simulated(platform):
    html = render_ad_preview_html(build_creative("CMP-1", platform, "Retargeting", "Lookalike"))
    assert SIMULATED_LABEL in html
    assert "data:image/svg+xml;base64," in html or platform == "Google"


def test_preview_escapes_untrusted_text():
    creative = dataclasses.replace(build_creative("CMP-2", "Facebook", "Conversions", "Lookalike"),
                                   headline="<script>alert(1)</script>", description='" onload="x')
    html = render_ad_preview_html(creative)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


@pytest.mark.parametrize("platform", list(ARTWORK_SIZES))
def test_artwork_is_valid_svg_with_requested_size(platform):
    width, height = ARTWORK_SIZES[platform]
    svg = render_artwork_svg(build_creative("CMP-3", platform, "Conversions", "Lookalike"), width, height)
    root = ET.fromstring(svg)
    assert root.attrib["width"] == str(width) and root.attrib["height"] == str(height)
