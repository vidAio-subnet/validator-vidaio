-- Exact upscaling output geometry (evaluation-item commitment v2).
--
-- A scale factor alone is not a sufficient output contract: the procedural
-- downscale applies a subpixel crop and even truncation, so input_width*factor
-- can be several pixels smaller than the pristine reference required by the
-- scoring geometry gate. New upscaling items persist and v2-commit both target
-- dimensions. Existing v1 rows remain NULL/NULL and stay independently
-- verifiable through the historical v1 commitment preimage.

ALTER TABLE evaluation_items ADD COLUMN target_width INTEGER
    CHECK (target_width IS NULL OR target_width > 0);
ALTER TABLE evaluation_items ADD COLUMN target_height INTEGER
    CHECK (target_height IS NULL OR target_height > 0);

CREATE TRIGGER evaluation_item_geometry_pair_insert
BEFORE INSERT ON evaluation_items
WHEN (NEW.target_width IS NULL) != (NEW.target_height IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'target_width and target_height must appear together');
END;

CREATE TRIGGER evaluation_item_geometry_pair_update
BEFORE UPDATE OF target_width, target_height ON evaluation_items
WHEN (NEW.target_width IS NULL) != (NEW.target_height IS NULL)
BEGIN
    SELECT RAISE(ABORT, 'target_width and target_height must appear together');
END;
