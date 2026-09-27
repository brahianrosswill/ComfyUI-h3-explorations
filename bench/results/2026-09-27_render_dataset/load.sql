-- Load the render dataset into DuckDB:  duckdb h3_renders.duckdb < load.sql
CREATE OR REPLACE TABLE renders  AS SELECT * FROM read_json_auto('renders.jsonl',  format='newline_delimited');
CREATE OR REPLACE TABLE measures AS SELECT * FROM read_json_auto('measures.jsonl', format='newline_delimited');
CREATE OR REPLACE TABLE models   AS SELECT * FROM read_json_auto('models.jsonl',   format='newline_delimited');
CREATE OR REPLACE TABLE findings AS SELECT * FROM read_json_auto('findings.jsonl', format='newline_delimited');
-- One row per render with the common measures as columns (valid looks only).
CREATE OR REPLACE VIEW looks AS
SELECT r.render_id, r.scene, r.arm, r.model_family, r.code_version,
       max(value) FILTER (WHERE metric='rms_contrast') AS contrast,
       max(value) FILTER (WHERE metric='white')        AS white,
       max(value) FILTER (WHERE metric='haze')         AS haze,
       max(value) FILTER (WHERE metric='sat')          AS sat,
       max(value) FILTER (WHERE metric='chroma')       AS chroma,
       max(value) FILTER (WHERE metric='hf')           AS hf,
       max(value) FILTER (WHERE metric='moved_share')  AS moved,
       max(value) FILTER (WHERE metric='boil')         AS boil,
       max(value) FILTER (WHERE metric='lufs')         AS lufs,
       max(value) FILTER (WHERE metric='centroid_hz')  AS audio_centroid_hz
FROM renders r JOIN measures m USING (render_id)
WHERE r.valid_look
GROUP BY ALL;
