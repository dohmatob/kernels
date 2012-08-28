cmake_minimum_required (VERSION 2.8 FATAL_ERROR)

project(kernels)

##-- version stuff
set(kernels_VERSION_CURRENT "0")
set(kernels_VERSION_REVISION "0")
set(kernels_VERSION_AGE "1")
set(kernels_VERSION "${kernels_VERSION_CURRENT}.${kernels_VERSION_REVISION}.${kernels_VERSION_AGE}")

if(UNIX)
  #-- set bin
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  
  #-- Boost
  find_package(Boost 1.42 COMPONENTS unit_test_framework REQUIRED)
  if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})
  endif(Boost_FOUND)

  include_directories(${CMAKE_SOURCE_DIR}/include)

  add_executable(Trie src/Trie.cxx)

  message("
   
    ====================================================================================
    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public License
    as published by the Free Software Foundation, either version 3 of
    the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program. If not, see
    <http://www.gnu.org/licenses/>.

    kernels-${kernels_VERSION}
    Copyright (c) 2012 dohmatob elvis dopgima
    ====================================================================================
    ")
  
else (UNIX)
  message("ONLY SUPPORTED FOR UNIX SYSTEMS!")
endif(UNIX)
