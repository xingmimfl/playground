/* -*- coding: utf-8 -*-
* Created by mengfanli on 2022/01/04
* code come from http://cristal.univ-lille.fr/~casiez/1euro/
*/

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <ctime>
#include "OneEuroFilter.hpp"
    // -----------------------------------------------------------------
    // Utilities
    
    
    void LowPassFilter::setAlpha(float alpha)
    {
        a = alpha ;
    }
    
    LowPassFilter::LowPassFilter(float alpha, float initval)
    {
        y = s = initval ;
        setAlpha(alpha) ;
        initialized = false ;
    }
    
    void LowPassFilter::reset(float value)
    {
        initialized = false;
        filter(value);
    }
    
    void LowPassFilter::clear()
    {
        y = s = 0.0 ;
        initialized = false;
    }
    
    float LowPassFilter::filter(float value)
    {
        float result ;
        if (initialized)
            result = a*value + (1.0-a)*s ;
        else
        {
            result = value ;
            initialized = true ;
        }
        y = value ;
        s = result ;
        return result ;
    }
    
    float LowPassFilter::filterWithAlpha(float value, float alpha)
    {
        setAlpha(alpha) ;
        return filter(value) ;
    }
    
    bool LowPassFilter::hasLastRawValue(void)
    {
        return initialized ;
    }
    
    float LowPassFilter::lastRawValue(void)
    {
        return y ;
    }
    
    float OneEuroFilter::alpha(float cutoff)
    {
        float te = 1.0 / freq_ ;
        float tau = 1.0 / (2*M_PI*cutoff) ;
        return 1.0 / (1.0 + tau/te) ;
    }
    
    void OneEuroFilter::setFrequency(float f)
    {
        freq_ = f ;
    }
    
    void OneEuroFilter::setMinCutoff(float mc)
    {
        mincutoff_ = mc ;
    }
    
    void OneEuroFilter::setBeta(float b)
    {
        beta_ = b ;
    }
    
    void OneEuroFilter::setDerivateCutoff(float dc)
    {
        dcutoff_ = dc ;
    }
    
    
    OneEuroFilter::OneEuroFilter()
    {
        x = nullptr;
        dx = nullptr;
    }
    OneEuroFilter::OneEuroFilter(float freq, float mincutoff, float beta, float dcutoff)
    {
        initFilter(freq, mincutoff, beta, dcutoff);
    }
    OneEuroFilter::~OneEuroFilter(void)
    {
        delete x;
        delete dx;
    }
    void OneEuroFilter::initFilter(float freq, float mincutoff, float beta, float dcutoff)
    {
        setFrequency(freq) ;
        setMinCutoff(mincutoff) ;
        setBeta(beta);
        setDerivateCutoff(dcutoff) ;
        x = new LowPassFilter(alpha(mincutoff)) ;
        dx = new LowPassFilter(alpha(dcutoff)) ;
        lasttime_ = UndefinedTime ;
    }
    
    float OneEuroFilter::filter(float value, TimeStamp timestamp)
    {
        // update the sampling frequency based on timestamps
        if (lasttime_!=UndefinedTime && timestamp!=UndefinedTime)
            freq_ = 1.0 / (timestamp-lasttime_) ;
        lasttime_ = timestamp ;
        // estimate the current variation per second
        float dvalue = x->hasLastRawValue() ? (value - x->lastRawValue())*freq_ : 0.0 ; // FIXME: 0.0 or value?
        edvalue_ = dx->filterWithAlpha(dvalue, alpha(dcutoff_)) ;
        // use it to update the cutoff frequency
        float cutoff = mincutoff_ + beta_*fabs(edvalue_) ;
        // filter the given value
        float result = x->filterWithAlpha(value, alpha(cutoff));
        return  result;
    }
    
    
    void OneEuroFilter::reset(float value)
    {
        x -> reset(value);
        dx -> reset(value);
    }
    
    void OneEuroFilter::clear()
    {
        x -> clear();
        dx -> clear();
    }
    
    float OneEuroFilter::getEdvalue()
    {
        return edvalue_ / freq_ + x -> lastRawValue();
    }
    
