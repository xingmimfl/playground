/* -*- coding: utf-8 -*-
* Created by mengfanli on 2022/01/04
* code come from http://cristal.univ-lille.fr/~casiez/1euro/
*/

#ifndef  one_euro_filter_hpp
#define one_euro_filter_hpp

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <ctime>

    typedef float TimeStamp ; // in seconds
    static const TimeStamp UndefinedTime = -1.0 ;

    //----LowPassFilter----
    class LowPassFilter
    {
    public:
        LowPassFilter(float alpha, float initval=0.0);
        void reset(float value);
        void clear();
        float filter(float value); // value: current value
        float filterWithAlpha(float value, float alpha);
        bool hasLastRawValue(void);
        float lastRawValue(void);
    protected:
        float y, a, s ;
        bool initialized ;
        void setAlpha(float alpha);
    };
    
    // OneEuro
    class OneEuroFilter
    {
    public:
        OneEuroFilter();
        OneEuroFilter(float freq, float mincutoff, float beta=0.0, float dcutoff=1.0);
        void initFilter(float freq = 1.0, float mincutoff=1.0, float beta=0.0, float dcutoff=1.0);
        float filter(float value, TimeStamp timestamp=UndefinedTime);
        void reset(float value);
        void clear();
        float getEdvalue();
        void setMinCutoff(float mc);
        ~OneEuroFilter();
        
        float alpha(float cutoff);
        void setFrequency(float f);
        void setBeta(float b);
        void setDerivateCutoff(float dc);
    protected:
        float freq_;
        float mincutoff_;
        float beta_ ;
        float dcutoff_;
        float edvalue_;
        LowPassFilter *x ;
        LowPassFilter *dx ;
        TimeStamp lasttime_;
    };
    
#endif /* smooth_filter_hpp */
