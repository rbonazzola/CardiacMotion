#!/bin/bash
gpg $HOME/.passwd-s3fs.gpg &> /dev/null
chmod 600 $HOME/.passwd-s3fs
s3fs $S3_BUCKET_NAME $S3_MOUNT_DIRECTORY
/bin/bash "$@"
