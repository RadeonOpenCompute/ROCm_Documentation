.. _Testing:

==========
Testing
==========

UCX uses Google Test Framework (https://github.com/google/googletest) This framework is intergated into project and does not require gtest preinstallation.

 * location: < root >/test/gtest
 * build: use --enable-gtest configuration option
 * launch:
 * make gtest
 * update Google Test Framework:
 * download latest stable version
 * launch fuse_gtest_files.py [GTEST_ROOT_DIR] OUTPUT_DIR from scripts folder
 * replace gtest/common/gtest.h and gtest/common/gtest-all.cc files with new from OUTPUT_DIR location.
