#include "CTrame.h"

#include "../../custom/custom_cuda.h"

CTrame::CTrame(int width, int height){
    _width        = width;
    _height       = height;
    _frame        = 1;
    t_in_bits     = new int   [ nb_info() ];
    t_coded_bits  = new int   [ nb_data() ];

    CUDA_MALLOC_HOST(&t_noise_data, nb_data() + 1);
//    t_noise_data  = new float[ nb_data() + 1 ];
    t_fpoint_data = new int   [ nb_data() ];
    t_decode_data = new int   [ nb_data() ];
    t_decode_bits = new int   [ nb_info() ];
}

CTrame::CTrame(int width, int height, int frame){
    _width        = width;
    _height       = height;
    _frame        = frame;
    t_in_bits     = new int   [ nb_info() * frame ];
    t_coded_bits  = new int   [ nb_data() * frame ];
    CUDA_MALLOC_HOST(&t_noise_data, nb_data()  * frame + 4);
//    t_noise_data  = new float[ nb_data() ];
    t_fpoint_data = new int   [ nb_data() * frame ];
    t_decode_data = new int   [ nb_data() * frame ];
    t_decode_bits = new int   [ nb_info() * frame ];
}

CTrame::~CTrame(){
    delete t_in_bits;
    delete t_coded_bits;
    //    delete t_noise_data;
	cudaFreeHost(t_noise_data);
    delete t_fpoint_data;
    delete t_decode_data;
    delete t_decode_bits;
}

int CTrame::nb_info(){
    return  /*nb_frames() * */(nb_data()-nb_checks());
}

int CTrame::nb_frames(){
    return  _frame;
}

int CTrame::nb_checks(){
    return _height;
}

int CTrame::nb_data(){
    return _width;
}

int* CTrame::get_t_in_bits(){
    return t_in_bits;
}

int* CTrame::get_t_coded_bits(){
    return t_coded_bits;
}

float* CTrame::get_t_noise_data(){
    return t_noise_data;
}

int* CTrame::get_t_fpoint_data(){
    return t_fpoint_data;
}

int* CTrame::get_t_decode_data(){
    return t_decode_data;
}

int* CTrame::get_t_decode_bits(){
    return t_decode_bits;
}
