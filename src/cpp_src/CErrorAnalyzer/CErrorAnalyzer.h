#ifndef CLASS_CError_Analyzer
#define CLASS_CError_Analyzer

#include "../CTrame/CTrame.h"

class CErrorAnalyzer
{
private:
	long int nb_bit_errors;
	long int nb_frame_errors;
	long int nb_analyzed_frames;
    int*     buf_en_bits;

protected:
    int  _vars;
    int  _data;
    int  _frames;
    int*  t_in_bits;
    int*  t_enc_bits;
    int*  t_decode_data;

    int _max_fe;
    bool _auto_fe_mode;
    
public:
    CErrorAnalyzer(CTrame *t);
    CErrorAnalyzer(CTrame *t, int max_fe);
    CErrorAnalyzer(CTrame *t, int max_fe, bool auto_fe_mode);
    virtual void generate();
    virtual void store_enc_bits();
    virtual void generate(int nErrors);

    long int nb_processed_frames();
    long int nb_fe();
    long int nb_be();

    long int fe_limit();
    bool fe_limit_achieved();

    double fer_value();
    double ber_value();

    long int nb_processed_frames(int add);
    long int nb_fe(int add);
    long int nb_be(int add);

    int nb_data();
    int nb_vars();
    int nb_checks();
};

#endif
