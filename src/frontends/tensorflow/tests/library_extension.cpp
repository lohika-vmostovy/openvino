// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "library_extension.hpp"

#include "tf_utils.hpp"

using namespace ov::frontend;

using TFLibraryExtensionTest = FrontendLibraryExtensionTest;

static FrontendLibraryExtensionTestParams getTestData() {
    FrontendLibraryExtensionTestParams params;
    params.m_frontEndName = TF_FE;
    params.m_modelsPath = std::string(TEST_TENSORFLOW_MODELS_DIRNAME);
    params.m_modelName = "2in_2out/2in_2out.pb";
    return params;
}

INSTANTIATE_TEST_SUITE_P(TFLibraryExtensionTest,
                         FrontendLibraryExtensionTest,
                         ::testing::Values(getTestData()),
                         FrontendLibraryExtensionTest::getTestCaseName);
