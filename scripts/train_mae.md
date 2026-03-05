# modify models/minkunet_attention.py
- current MinkUNetSparseAttention is Dense (torch.tensor) -> Dense, I want to segment it into 3 networks: MinkUNetSparseAttention_Input (Dense to Sparse), MinkUNetSparseAttentionCore (Sparse (warpconv.net.Voxels) to Sparse) and MinkUNetSparseAttention_Output (Sparse to Dense). And add a wrapper with current name "MinkUNetSparseAttention" to keep back compatibility.

# Build a training workflow for Sparse Masked Auto Encoder
- implement a sparse masking algorithm
  - from a warpconvnet.Voxels (COO sparse format)
  - 3 configurable parameters: `masking_frac` (0-1), `win_ch`(int) and `win_tick`(int)
  - randomlly sample masking_frac points in Voxels. Then opne up a window with `2*win_ch` in the channel direction and `2*win_tick` in the tick direction. Find all the active points in that window and set them to 0.
  - note keep the original warpconvnet.Voxels mask for the `target_spatial_sparse_tensor` so that we know which 0 points were active before masking out.
- implement the self-supervised learn (SSL) training in `scripts/train_mae.py`
  - reference this script: training.py
  - example input data is here: `/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27`, the h5 contains sparse data
  - use this `dataset`: `loader/apa_sparse_dataset.py`
  - use the previous masking algorithm, feed the masked warpconvnet.Voxels to the backbone.
  - use the `MinkUNetSparseAttentionCore` as the backbone, so directly use the sparse input and output a sparse tensor (warpconvnet.Voxels) with (B, N, 2) vox.coordinate_tensor and (B, N, Feature) vox.feature_tensor
  - attach a sparse CNN head `charge_head` to convert the previous feature (B, N, Feature) to (B, N, 1) to match the un-masked original sparse tensor
  - use point-wise L1 loss.
- implement a downstream `nu_flavor` head for a supervised fine tune (SFT) learning
  - update the dataset `models/minkunet_attention.py` to readin the meta data. e.g., this file `/nfs/data/1/yuhw/cffm-data/prod-jay-1M-2026-02-27-sample/13717/1/001/out_monte-carlo-013717-000001_310694_6_1_20260224T033536Z/monte-carlo-013717-000001_310694_6_1_20260224T033536Z_metadata.h5`
  - the `nu_flavor` head use the feature from the backbone ((B, N, Feature)) and convert it to (B, nClass). nClass is the number of `nu_pdg`
  - when training the `nu_flavor` head, freeze the backbone
  - use approporate loss like the cross-entropy
- do a interleafed SSL and SFT to understand the how useful the features obtained from SSL.
  - 2 configurable numbers, `nSSL`, `nSFT` to indicate that do SSL for `nSSL` batches, then do `nSFT` batches
  - note the SFT should not report any information to the SSL
- monitor these below
  - SSL loss
  - SFT loss
  - SFT confusion matrix
  - SFT per `nu_pdg` class efficiency and purity