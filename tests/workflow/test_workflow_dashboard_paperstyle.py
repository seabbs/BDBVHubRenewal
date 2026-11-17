"""A reference can take its model's colour in the paper figures."""

import pytest

pytest.importorskip("matplotlib")

from flowlib.dashboard import paperstyle


def test_reference_models_take_their_paper_colours():
    from paperfigs import style

    assert (
        paperstyle.model_colour("randomwalknowcast")
        == style.model_style("randomwalknowcast", ["randomwalknowcast"])[0]
    )
