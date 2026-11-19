# Data and weights

Obtain FLIR ADAS V2 from the [official FLIR dataset page](https://oem.flir.com/solutions/automotive/adas-dataset-form/) and follow the terms supplied with your download. This repository does not grant rights to the dataset.

No source images, converted images, annotations, qualitative dataset crops, checkpoints or pretrained weights are included. The research workspace did not establish an explicit dataset redistribution grant. The release therefore includes code and configurations only.

The FLIR converter preserves the official train/validation separation. The nine-class mapping is: person, bike, car, motor, bus, truck, light, hydrant, sign. Review its conversion summary before using the converted dataset. Keep raw files immutable and write conversion outputs to a new directory.

The frozen source normalization is included as a small JSON configuration. Per-image source features, experiment results and the original M3FD split manifest are omitted from this compact release.

Exact replication of the original detector outputs requires the original checkpoints and dataset version. The six training configurations allow retraining, but do not guarantee bitwise-identical weights across software or hardware versions. Checkpoint download links are not yet provided.
