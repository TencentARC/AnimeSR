# Some notes about calculating metrics

* For fast evaluation, we measure 1 frames every 10 frames, that is, the 0-th frame, 10-th frame, 20-th frame, etc. will participate the metrics calculation.
* MANIQA
  * codebase: clone the official github [repo](https://github.com/IIGROUP/MANIQA), put the [inference script](inference_MANIQA.py) into the project root
  * **checkpoint**: we use the ensemble model provided by the authors to compute MANIQA. You can download the checkpoint from [Ondrive](https://1drv.ms/u/s!Akkg8_btnkS7mU7eb_dFDHkW05B2?e=G922hL).
  * note: the result has certain randomness, but the error should be relatively small.
