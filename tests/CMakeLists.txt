add_executable(ctd_tests
  test_smoke.cpp
  test_tensor.cpp
  test_elementwise.cpp
  test_activations.cpp
  test_embedding.cpp
  test_matmul.cpp
  test_bmm.cpp
  test_norm.cpp
  test_softmax.cpp
  test_rope.cpp
  test_permute.cpp
  test_mask.cpp
  test_integration_embedding.cpp
  test_integration_block.cpp
  test_integration_model.cpp
  test_integration_kvcache.cpp
  test_autograd.cpp
  test_train_forward.cpp
  test_data_bins.cpp
)

# Integration tests need to locate reference artifacts under the repo.
target_compile_definitions(ctd_tests PRIVATE
  CTD_REPO_ROOT="${CMAKE_SOURCE_DIR}"
)

target_link_libraries(ctd_tests
  PRIVATE
    ctd
    GTest::gtest_main
    ctd_warnings
)

include(GoogleTest)
gtest_discover_tests(ctd_tests)
