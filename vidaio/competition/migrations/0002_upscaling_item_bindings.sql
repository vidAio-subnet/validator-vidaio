-- Backward-compatible expansion for upscaling competitions.
--
-- Existing rows can only be compression rows because manifest v1 rejected every
-- other track.  Normalize their reference to the miner input while leaving the
-- upscaling-only factor/commitment NULL.  New upscaling rows are validated against
-- the pre-enrollment manifest by repository.add_evaluation_item before insertion.

ALTER TABLE evaluation_items ADD COLUMN reference_sha256 TEXT;
ALTER TABLE evaluation_items ADD COLUMN reference_bytes INTEGER
    CHECK (reference_bytes IS NULL OR reference_bytes >= 0);
ALTER TABLE evaluation_items ADD COLUMN upscale_factor INTEGER
    CHECK (upscale_factor IS NULL OR upscale_factor IN (2, 4));
ALTER TABLE evaluation_items ADD COLUMN item_commitment TEXT
    CHECK (item_commitment IS NULL OR
           (typeof(item_commitment) = 'text'
            AND length(item_commitment) = 64
            AND item_commitment NOT GLOB '*[^0-9a-f]*'));

UPDATE evaluation_items
SET reference_sha256 = input_sha256,
    reference_bytes = input_bytes
WHERE reference_sha256 IS NULL;

-- Holdout/input bytes are single-use across competitions.  This is deliberately
-- cross-kind: a future pristine reference cannot reuse an earlier miner input, and
-- vice versa.  Repository validation provides a readable error; these triggers are
-- the concurrency/direct-SQL backstop.
CREATE TRIGGER evaluation_media_single_competition_insert
BEFORE INSERT ON evaluation_items
WHEN EXISTS (
    SELECT 1 FROM evaluation_items AS old
    WHERE old.competition_id != NEW.competition_id
      AND (
          old.input_sha256 = NEW.input_sha256
          OR old.reference_sha256 = NEW.input_sha256
          OR (NEW.reference_sha256 IS NOT NULL
              AND old.input_sha256 = NEW.reference_sha256)
          OR (NEW.reference_sha256 IS NOT NULL
              AND old.reference_sha256 = NEW.reference_sha256)
      )
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation media digests are single-use across competitions');
END;

CREATE TRIGGER evaluation_media_single_competition_update
BEFORE UPDATE OF competition_id, input_sha256, reference_sha256 ON evaluation_items
WHEN EXISTS (
    SELECT 1 FROM evaluation_items AS old
    WHERE old.item_id != NEW.item_id
      AND old.competition_id != NEW.competition_id
      AND (
          old.input_sha256 = NEW.input_sha256
          OR old.reference_sha256 = NEW.input_sha256
          OR (NEW.reference_sha256 IS NOT NULL
              AND old.input_sha256 = NEW.reference_sha256)
          OR (NEW.reference_sha256 IS NOT NULL
              AND old.reference_sha256 = NEW.reference_sha256)
      )
)
BEGIN
    SELECT RAISE(ABORT, 'evaluation media digests are single-use across competitions');
END;
