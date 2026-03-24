#ifndef SPECIALIZATION_CUH
#define SPECIALIZATION_CUH

#include <string>

typedef struct CUctx_st *CUcontext;

namespace warpscope {
namespace specialization {

void init(const std::string &json_path);
void on_kernel_exit();
void term(CUcontext ctx);

} // namespace specialization
} // namespace warpscope

#endif // SPECIALIZATION_CUH
