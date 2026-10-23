"""Artist attribution (`Drawn by <artist>`) in caption building, and its cache-key behaviour."""

import random
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

from trainer.data.caption import CaptionConfig, build_caption, caption_identity
from trainer.data.caption_variations import digest
from trainer.data.text_cache import validate_static_captions

TAGS = "1girl, Drawn by emily (pure dream), blue hair, Drawn by alice, smile, outdoors, sky, tree"
NL = ("A girl stands under a tree. Drawn by emily (pure dream). The sky is clear. "
      "She is smiling.")
ATTR = ["^drawn by\\s"]


def tags_of(caption, delimiter=", "):
    return caption.split(delimiter)


class AttributionTests(unittest.TestCase):
    def cfg(self, **kw):
        return CaptionConfig(attribution_patterns=ATTR, **kw)

    def test_fixed_pins_every_artist_in_order_and_survives_full_dropout(self):
        cfg = self.cfg(shuffle_tags=True, tag_dropout_percent=1.0, min_tags_kept=0)
        for seed in range(50):
            out = tags_of(build_caption(TAGS, None, cfg, random.Random(seed)))
            self.assertEqual(out[:2], ["Drawn by emily (pure dream)", "Drawn by alice"])
            # Dropout 1.0 removed every ordinary tag; only the attributions are immune.
            self.assertEqual(len(out), 2)

    def test_min_tags_kept_counts_the_attributions(self):
        cfg = self.cfg(tag_dropout_percent=1.0, min_tags_kept=4)
        out = tags_of(build_caption(TAGS, None, cfg, random.Random(0)))
        self.assertEqual(len(out), 4)

    def test_random_position_keeps_exactly_one_copy_each(self):
        cfg = self.cfg(shuffle_tags=True, attribution_position="random")
        positions = set()
        for seed in range(200):
            out = tags_of(build_caption(TAGS, None, cfg, random.Random(seed)))
            self.assertEqual(out.count("Drawn by alice"), 1)
            self.assertEqual(out.count("Drawn by emily (pure dream)"), 1)
            positions.add(out.index("Drawn by alice"))
        self.assertGreater(len(positions), 4)

    def test_not_immune_can_be_dropped_but_fixed_still_pins_survivors(self):
        cfg = self.cfg(tag_dropout_percent=0.5, attribution_dropout_immune=False,
                       shuffle_tags=True)
        dropped = False
        for seed in range(100):
            out = tags_of(build_caption(TAGS, None, cfg, random.Random(seed)))
            survivors = [t for t in out if t.startswith("Drawn by")]
            dropped |= len(survivors) < 2
            self.assertEqual(out[:len(survivors)], survivors)
        self.assertTrue(dropped)

    def test_nl_attribution_pinned_and_shuffle_keeps_single_periods(self):
        cfg = self.cfg(caption_mode="nl", nl_shuffle_sentences=True)
        for seed in range(30):
            out = build_caption(TAGS, NL, cfg, random.Random(seed))
            self.assertTrue(out.startswith("Drawn by emily (pure dream). "))
            self.assertNotIn("..", out)
            self.assertTrue(out.endswith("."))

    def test_tags_nl_dedupes_only_what_the_tags_already_carry(self):
        cfg = self.cfg(caption_mode="tags_nl")
        out = build_caption(TAGS, NL, cfg, random.Random(0))
        self.assertEqual(out.count("Drawn by emily (pure dream)"), 1)
        self.assertEqual(out.count("Drawn by alice"), 1)
        # An attribution only the NL names is kept: dedupe must never delete the last copy.
        out = build_caption("1girl, smile", NL, cfg, random.Random(0))
        self.assertEqual(out.count("Drawn by emily (pure dream)"), 1)
        no_dedupe = replace(cfg, attribution_dedupe_on_combine=False)
        self.assertEqual(build_caption(TAGS, NL, no_dedupe, random.Random(0))
                         .count("Drawn by emily (pure dream)"), 2)

    def test_nl_tags_dedupes_the_tag_side(self):
        cfg = self.cfg(caption_mode="nl_tags")
        out = build_caption(TAGS, NL, cfg, random.Random(0))
        self.assertTrue(out.startswith("Drawn by emily (pure dream). A girl"))
        self.assertEqual(out.count("Drawn by emily (pure dream)"), 1)
        # alice is credited only in the tags, so it stays.
        self.assertEqual(out.count("Drawn by alice"), 1)

    def test_off_is_untouched(self):
        plain = CaptionConfig(shuffle_tags=True, tag_dropout_percent=0.3)
        for seed in range(20):
            a = build_caption(TAGS, NL, plain, random.Random(seed))
            b = build_caption(TAGS, NL, replace(plain, attribution_position="random"),
                              random.Random(seed))
            self.assertEqual(a, b)

    def test_cache_identity_is_unchanged_while_off(self):
        # Existing caption-variation caches key on the config; adding fields must not move them.
        plain = CaptionConfig(shuffle_tags=True)
        legacy = asdict(plain)
        for key in ("attribution_patterns", "attribution_position",
                    "attribution_dropout_immune", "attribution_dedupe_on_combine",
                    "keep_order_through_attribution", "annotation_attribution_template", "layout_item_dropout", "layout_panels", "layout_panels_as_objects", "layout_short_captions", "layout_parts",
                    "protected_tags_file"):   # a machine-specific path, never part of the key
            legacy.pop(key)
        self.assertEqual(digest(caption_identity(plain)), digest(legacy))
        self.assertNotEqual(digest(caption_identity(self.cfg(shuffle_tags=True))), digest(legacy))

    def test_static_cache_rejects_random_position(self):
        def config(caption, variations=0):
            return SimpleNamespace(dataset=SimpleNamespace(caption=caption),
                                   curriculum=SimpleNamespace(phases=[]),
                                   preserve=SimpleNamespace(enabled=False),
                                   train=SimpleNamespace(caption_variations=variations))
        validate_static_captions(config(self.cfg()))
        with self.assertRaisesRegex(ValueError, "attribution_position=random"):
            validate_static_captions(config(self.cfg(attribution_position="random")))
        validate_static_captions(config(self.cfg(attribution_position="random"), variations=4))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "invalid regex"):
            CaptionConfig(attribution_patterns=["(unclosed"])
        with self.assertRaisesRegex(ValueError, "fixed' or 'random"):
            CaptionConfig(attribution_patterns=ATTR, attribution_position="front")
        self.assertEqual(CaptionConfig(attribution_patterns="^by ").attribution_patterns, ["^by "])


if __name__ == "__main__":
    unittest.main()
