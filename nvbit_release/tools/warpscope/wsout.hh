#ifndef WSOUT_HH
#define WSOUT_HH

#include <iostream>
#include <string>

// Return the current logging stream with "#warpscope: " prefix.
std::ostream& wsout();

// Return the underlying stream WITHOUT the prefix.
std::ostream& wsout_stream();

// Redirect output to a file.
void set_out_file(const std::string& filename);

#endif // WSOUT_HH
