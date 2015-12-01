#include "CErrorAnalyzer.h"

CErrorAnalyzer::CErrorAnalyzer(CTrame *t){
    _data         = t->nb_data();
    _vars         = t->nb_info();
    _frames       = t->nb_frames();

    t_decode_data = t->get_t_decode_data();
    t_in_bits     = t->get_t_in_bits();
    t_enc_bits    = t->get_t_coded_bits();

    buf_en_bits   = new int [_data * _frames];

    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
    _max_fe            = 200;
    _auto_fe_mode      = false;
}

CErrorAnalyzer::CErrorAnalyzer(CTrame *t, int max_fe){
    _data              = t->nb_data();
    _vars              = t->nb_info();
    _frames            = t->nb_frames();

    t_decode_data      = t->get_t_decode_data();
    t_in_bits          = t->get_t_in_bits();
    t_enc_bits         = t->get_t_coded_bits();

    buf_en_bits        = new int [_data * _frames];

    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
    _max_fe            = max_fe;
    _auto_fe_mode      = false;
}

CErrorAnalyzer::CErrorAnalyzer(CTrame *t, int max_fe, bool auto_fe_mode){
    _data              = t->nb_data();
    _vars              = t->nb_info();
    _frames            = t->nb_frames();

    t_decode_data      = t->get_t_decode_data();
    t_in_bits          = t->get_t_in_bits();
    t_enc_bits         = t->get_t_coded_bits();

    buf_en_bits        = new int [_data * _frames];

    nb_bit_errors      = 0;
    nb_frame_errors    = 0;
    nb_analyzed_frames = 0;
    _max_fe            = max_fe;
    _auto_fe_mode      = auto_fe_mode;
}

long int CErrorAnalyzer::fe_limit()
{
    if( _auto_fe_mode == false ){
        return _max_fe;
    }else{
        double tBER = ber_value();
        if( tBER < 1.0e-9){
            return (_max_fe/16);
        }else if( tBER < 1.0e-8){
            return (_max_fe/8);
        }else if( tBER < 1.0e-7){
            return (_max_fe/4);
        }else if( tBER < 1.0e-6){
            return (_max_fe/2);
        }else{
            return (_max_fe);
        }
    }
}

bool CErrorAnalyzer::fe_limit_achieved()
{
    return (nb_fe() >= fe_limit());
}

void CErrorAnalyzer::generate(){

	for(int k=0; k<_frames; k++){
		int nErrors = 0;
	    for(int i=0; i<_data; i++){
	        nErrors += (t_decode_data[k*_data+i] != buf_en_bits[k*_data+i]);
//	        if( t_decode_data[k * _data + i] != t_decode_data[i] ) exit( 0 );
	    }
//	    if( nErrors ) printf("(EE) Error in frame %d (%d errors)\n", k+1, nErrors);
	    nb_bit_errors      += nErrors;
	    nb_frame_errors    += (nErrors != 0);
	    nb_analyzed_frames += 1;
	}
}

void CErrorAnalyzer::generate(int nErrors){
    nb_bit_errors      += nErrors;
    nb_frame_errors    += (nErrors != 0);
    nb_analyzed_frames += 1;
}

void CErrorAnalyzer::store_enc_bits(){
	for(int k=0; k<_frames; k++){
	    for(int i=0; i<_data; i++){
	        buf_en_bits[k * _data + i] = t_enc_bits[k * _data + i];
	        if( buf_en_bits[k * _data + i] != t_enc_bits[i] ) exit( 0 );
	    }
	}

//    for(int i=0; i<_vars; i++){
//        buf_en_bits[i] = t_in_bits[i];
//    }
}

long int CErrorAnalyzer::nb_processed_frames(){
    return nb_analyzed_frames;
}

long int CErrorAnalyzer::nb_fe(){
    return nb_frame_errors;
}

long int CErrorAnalyzer::nb_be(){
    return nb_bit_errors;
}

double CErrorAnalyzer::fer_value(){
    double tFER = (((double)nb_fe())/(nb_processed_frames()));
    return tFER;
}

double CErrorAnalyzer::ber_value(){
    double tBER = (((double)nb_be())/(nb_processed_frames())/(_data));
    return tBER;
}

long int CErrorAnalyzer::nb_be(int data){
    nb_bit_errors = data;
    return nb_bit_errors;
}

long int CErrorAnalyzer::nb_processed_frames(int data){
    nb_analyzed_frames = data;
    return nb_analyzed_frames;
}

long int CErrorAnalyzer::nb_fe(int data){
    nb_frame_errors = data;
    return nb_frame_errors;
}

int CErrorAnalyzer::nb_data(){
    return _data;
}

int CErrorAnalyzer::nb_vars(){
    return _vars;
}

int CErrorAnalyzer::nb_checks(){
    return (_data - _vars);
}



