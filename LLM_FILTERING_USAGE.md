# LLM Filtering Usage

## New Argument: `--filter_every_n_frames`

Controls how often LLM filtering is applied during tracking.

### Options

- `--filter_every_n_frames 0` - **First frame only** (fastest, least accurate)
  - LLM runs once on frame 0
  - Selected IDs are used for entire video
  - Best for: Static scenes, computational constraints

- `--filter_every_n_frames 1` - **Every frame** (slowest, most accurate)
  - LLM runs on every single frame
  - Most accurate but very slow
  - Best for: Dynamic scenes, short videos

- `--filter_every_n_frames N` - **Every N frames** (balanced)
  - LLM runs on frames 0, N, 2N, 3N, etc.
  - Between filtered frames, uses previous selection
  - Best for: Most practical scenarios

### Examples

```bash
# First frame only (original behavior)
python bytetrack_inference.py image \
  --filter_every_n_frames 0 \
  --prompt "red car"

# Every frame (slowest, most accurate)
python bytetrack_inference.py image \
  --filter_every_n_frames 1 \
  --prompt "red car"

# Every 5 frames (good balance)
python bytetrack_inference.py image \
  --filter_every_n_frames 5 \
  --prompt "red car"

# Every 10 frames (faster, still adaptive)
python bytetrack_inference.py image \
  --filter_every_n_frames 10 \
  --prompt "red car"
```

### Performance Impact

Assuming 100 frames, 2 seconds per LLM call:

| Setting | LLM Calls | Total Time | Use Case |
|---------|-----------|------------|----------|
| `0` | 1 | 2 sec | Static scenes |
| `10` | 10 | 20 sec | Good balance |
| `5` | 20 | 40 sec | More adaptive |
| `1` | 100 | 200 sec | Maximum accuracy |

### How It Works

1. **On filtered frames** (0, N, 2N, ...):
   - Save crops of all detected objects
   - Send crops to LLM with prompt
   - Get new selected_target_ids

2. **On non-filtered frames**:
   - Use selected_target_ids from last filtering
   - No LLM calls = faster processing

3. **Filtering logic**:
   - Only tracks with IDs in selected_target_ids pass through
   - Others are ignored in the output

### Recommendations

- **Short videos (<50 frames)**: Use `--filter_every_n_frames 1`
- **Medium videos (50-200 frames)**: Use `--filter_every_n_frames 5` or `10`
- **Long videos (>200 frames)**: Use `--filter_every_n_frames 10` or `20`
- **Very fast changes**: Lower N value
- **Stable tracking**: Higher N value
