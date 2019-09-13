### Developer options ###

option(KMEANS_TIDY "Run clang-tidy when building.")
option(KMEANS_SANITIZERS "Build with sanitizers.")
option(KMEANS_WARNINGS_AS_ERRORS "Add -Werror or equivalent to the compile flags and clang-tidy.")

mark_as_advanced(
  KMEANS_TIDY
  KMEANS_SANITIZERS
  KMEANS_WARNINGS_AS_ERRORS
)

### clang-tidy ###

if(KMEANS_TIDY)
  find_program(KMEANS_TIDY_PROGRAM clang-tidy)
  mark_as_advanced(KMEANS_TIDY_PROGRAM)

  if(KMEANS_TIDY_PROGRAM)
    if(KMEANS_WARNINGS_AS_ERRORS)
      set(KMEANS_TIDY_PROGRAM ${KMEANS_TIDY_PROGRAM} -warnings-as-errors=*)
    endif()
  else()
    message(FATAL_ERROR "clang-tidy not found")
  endif()
endif()

### Global Setup ###

if(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(/permissive- KMEANS_HAVE_PERMISSIVE)
endif()

if(KMEANS_SANITIZERS)
  if(MSVC)
    message(FATAL_ERROR "Building with sanitizers is not supported when using the Visual C++ toolchain.")
  endif()

  if(NOT ${CMAKE_CXX_COMPILER_ID} MATCHES GNU|Clang)
    message(FATAL_ERROR "Building with sanitizers is not supported when using the ${CMAKE_CXX_COMPILER_ID} compiler.")
  endif()
endif()

### Includes ###

include(GenerateExportHeader)
include(CMakePackageConfigHelpers)

### Functions ###

function(kmeans_add_common TARGET OUTPUT_DIRECTORY)
  target_compile_features(${TARGET} PUBLIC cxx_std_11)

  set_target_properties(${TARGET} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}"
    ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}"
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIRECTORY}"
    INTERPROCEDURAL_OPTIMIZATION TRUE
  )

  if(KMEANS_TIDY AND KMEANS_TIDY_PROGRAM)
    set_target_properties(${TARGET} PROPERTIES
      # `KMEANS_TIDY_PROGRAM` is a list so we surround it with quotes to pass it
      # as a single argument.
      CXX_CLANG_TIDY "${KMEANS_TIDY_PROGRAM}"
    )
  endif()

  ### Common development flags (warnings + sanitizers + colors) ###

  if(MSVC)
    target_compile_options(${TARGET} PRIVATE
      /nologo # Silence MSVC compiler version output.
      /wd4068 # Allow unknown pragmas.
      /wd4221 # Results in false positives.
      /arch:AVX
      /fp:fast
      $<$<BOOL:${KMEANS_WARNINGS_AS_ERRORS}>:/WX> # -Werror
      $<$<BOOL:${KMEANS_CXX_HAVE_PERMISSIVE}>:/permissive->
    )

    if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.15.0")
      # CMake 3.15 does not add /W3 to the compiler flags by default anymore
      # so we add /W4 instead.
      target_compile_options(${TARGET} PRIVATE /W4)
    endif()

    target_link_options(${TARGET} PRIVATE
      /nologo # Silence MSVC linker version output.
    )
  else()
    target_compile_options(${TARGET} PRIVATE
      -Wall
      -Wextra
      -pedantic
      -Wconversion
      -Wsign-conversion
      -Wno-unknown-pragmas
      $<$<BOOL:${KMEANS_WARNINGS_AS_ERRORS}>:-Werror>
      $<$<BOOL:${KMEANS_WARNINGS_AS_ERRORS}>:-pedantic-errors>
    )

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      target_compile_options(${TARGET} PRIVATE -xCORE-AVX-I)
    else()
      target_compile_options(${TARGET} PRIVATE -march=core-avx-i -ffast-math)
    endif()
  endif()

  if(KMEANS_SANITIZERS)
    target_compile_options(${TARGET} PRIVATE
      -fsanitize=address,undefined
    )
    target_link_options(${TARGET} PRIVATE
      -fsanitize=address,undefined
      # GCC sanitizers only work when using the gold linker.
      $<$<CXX_COMPILER_ID:GNU>:-fuse-ld=gold>
    )
  endif()

  target_compile_options(${TARGET} PRIVATE
    $<$<CXX_COMPILER_ID:GNU>:-fdiagnostics-color>
    $<$<CXX_COMPILER_ID:Clang>:-fcolor-diagnostics>
  )
endfunction()

function(kmeans_add_executable TARGET)
  add_executable(${TARGET} ${ARGN})
  kmeans_add_common(${TARGET} bin)
  target_include_directories(${TARGET} PRIVATE src)
endfunction()

function(kmeans_add_library TARGET)
  add_library(${TARGET} ${ARGN})
  kmeans_add_common(${TARGET} bin)
  target_include_directories(${TARGET} PRIVATE src)
endfunction()
