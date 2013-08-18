add_executable(test test.cxx)
target_link_libraries(test Trie)

# We need to include the unit test framework in the list of requirements in the find_package command, and then link it:
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
target_link_libraries(test
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )
