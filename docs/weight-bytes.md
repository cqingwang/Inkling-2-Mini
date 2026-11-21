# Weight bytes (from the safetensors headers, revision ed40e7da)

- 24 stored layers; layers 6-17 run 3 passes -> 48 layer applications per forward.
- Dense BF16 per layer (attn gate_attn/wq_du/wo_ud 14.2 MB each, wk_dv/wv_dv 4.7 MB, wr_du 1.8 MB, router gate 0.6 MB,
  controllers ~2.4 MB, sconv/rel small): 54.5 MB/layer -> 2,787 MB per forward (48 applications).
- LM head (unembed) BF16 201024 x 2304: 926 MB. Embedding 926 MB (lookup only).
- Experts: layers 0-19 NVFP4 (U8 packed + F8_E4M3 scale per 16): 4.48 MB per expert (w13 2304x2304, w2 1152x2304),
  573 MB per layer. Layers 20-23 MXFP8 (F8_E4M3 + U8 e8m0 scale per 32): 8.21 MB per expert, 1,051 MB per layer.
  Applications: NVFP4 44 (layers 0-5, 3x 6-17, 18-19), MXFP8 4. MoE bytes per forward = u * (44*4.48 + 4*8.21) MB
  = u * 230 MB, where u = distinct experts touched per layer application (union over the rows of the forward).
- DSpark draft (mtp.safetensors): 636 MB BF16 total (3 layers; markov head with markov_w1 201024x256 = 103 MB);
  draft logits presumably go through the target unembed (926 MB) - check in the trace.
- GB10: ~273 GB/s theoretical, ~220 GB/s achievable.
