# Instruction for calculating metrics

## Prepare the frames
For fast evaluation, we measure 1 frames every 10 frames, that is, the 0-th frame, 10-th frame, 20-th frame, etc. will participate the metrics calculation.
This can be achieved by `sample_interval` argument.
```bash
python scripts/inference_animesr_frames.py -i AVC-RealLQ-ROOT -n AnimeSR_v1-PaperModel --expname animesr_v1_si10 --sample_interval 10
```

## MANIQA calculation
### requirements
`pip install timm==0.5.4`
### checkpoint
we use the ensemble model provided by the authors to compute MANIQA. You can download the checkpoint from [Ondrive](https://1drv.ms/u/s!Akkg8_btnkS7mU7eb_dFDHkW05B2?e=G922hL).
### inference:
```bash
# cd into scripts/metrics/MANIQA
python inference_MANIQA.py --model_path MANIQA_CKPT_YOU_JUST_DOWNLOADED --input_dir ../../../results/animesr_v1_si10/frames --output_dir output/ensemble_attentionIQA2_finetune_e2/AnimeSR_v1_si10
```
note that the result has certain randomness, but the error should be relatively small.
### license
the MANIQA codes&checkpoint are original from [MANIQA](https://github.com/IIGROUP/MANIQA) and @[TianheWu](https://github.com/TianheWu).
