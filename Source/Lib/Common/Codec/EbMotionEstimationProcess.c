/*
* Copyright(c) 2019 Intel Corporation
* SPDX - License - Identifier: BSD - 2 - Clause - Patent
*/

#include <stdlib.h>

#include "EbUtility.h"
#include "EbPictureControlSet.h"
#include "EbSequenceControlSet.h"
#include "EbPictureDecisionResults.h"
#include "EbMotionEstimationProcess.h"
#include "EbMotionEstimationResults.h"
#include "EbReferenceObject.h"
#include "EbMotionEstimation.h"
#include "EbIntraPrediction.h"
#include "EbLambdaRateTables.h"
#include "EbComputeSAD.h"

#include "emmintrin.h"

#if ALTREF_FILTERING_SUPPORT
#include "EbTemporalFiltering.h"
#endif

/* --32x32-
|00||01|
|02||03|
--------*/
/* ------16x16-----
|00||01||04||05|
|02||03||06||07|
|08||09||12||13|
|10||11||14||15|
----------------*/
/* ------8x8----------------------------
|00||01||04||05|     |16||17||20||21|
|02||03||06||07|     |18||19||22||23|
|08||09||12||13|     |24||25||28||29|
|10||11||14||15|     |26||27||30||31|

|32||33||36||37|     |48||49||52||53|
|34||35||38||39|     |50||51||54||55|
|40||41||44||45|     |56||57||60||61|
|42||43||46||47|     |58||59||62||63|
-------------------------------------*/
EbErrorType CheckZeroZeroCenter(
    PictureParentControlSet   *picture_control_set_ptr,
    EbPictureBufferDesc        *refPicPtr,
    MeContext                  *context_ptr,
    uint32_t                       sb_origin_x,
    uint32_t                       sb_origin_y,
    uint32_t                       sb_width,
    uint32_t                       sb_height,
    int16_t                       *x_search_center,
    int16_t                       *y_search_center,
    EbAsm                       asm_type);

/************************************************
 * Set ME/HME Params from Config
 ************************************************/
void* set_me_hme_params_from_config(
    SequenceControlSet        *sequence_control_set_ptr,
    MeContext                 *me_context_ptr)
{
    uint16_t hmeRegionIndex = 0;

    me_context_ptr->search_area_width = (uint8_t)sequence_control_set_ptr->static_config.search_area_width;
    me_context_ptr->search_area_height = (uint8_t)sequence_control_set_ptr->static_config.search_area_height;

    me_context_ptr->number_hme_search_region_in_width = (uint16_t)sequence_control_set_ptr->static_config.number_hme_search_region_in_width;
    me_context_ptr->number_hme_search_region_in_height = (uint16_t)sequence_control_set_ptr->static_config.number_hme_search_region_in_height;

    me_context_ptr->hme_level0_total_search_area_width = (uint16_t)sequence_control_set_ptr->static_config.hme_level0_total_search_area_width;
    me_context_ptr->hme_level0_total_search_area_height = (uint16_t)sequence_control_set_ptr->static_config.hme_level0_total_search_area_height;

    for (hmeRegionIndex = 0; hmeRegionIndex < me_context_ptr->number_hme_search_region_in_width; ++hmeRegionIndex) {
        me_context_ptr->hme_level0_search_area_in_width_array[hmeRegionIndex] = (uint16_t)sequence_control_set_ptr->static_config.hme_level0_search_area_in_width_array[hmeRegionIndex];
        me_context_ptr->hme_level1_search_area_in_width_array[hmeRegionIndex] = (uint16_t)sequence_control_set_ptr->static_config.hme_level1_search_area_in_width_array[hmeRegionIndex];
        me_context_ptr->hme_level2_search_area_in_width_array[hmeRegionIndex] = (uint16_t)sequence_control_set_ptr->static_config.hme_level2_search_area_in_width_array[hmeRegionIndex];
    }

    for (hmeRegionIndex = 0; hmeRegionIndex < me_context_ptr->number_hme_search_region_in_height; ++hmeRegionIndex) {
        me_context_ptr->hme_level0_search_area_in_height_array[hmeRegionIndex] = (uint16_t)sequence_control_set_ptr->static_config.hme_level0_search_area_in_height_array[hmeRegionIndex];
        me_context_ptr->hme_level1_search_area_in_height_array[hmeRegionIndex] = (uint16_t)sequence_control_set_ptr->static_config.hme_level1_search_area_in_height_array[hmeRegionIndex];
        me_context_ptr->hme_level2_search_area_in_height_array[hmeRegionIndex] = (uint16_t)sequence_control_set_ptr->static_config.hme_level2_search_area_in_height_array[hmeRegionIndex];
    }

    return EB_NULL;
}

/************************************************
 * Set ME/HME Params
 ************************************************/
void* set_me_hme_params_oq(
    MeContext                     *me_context_ptr,
    PictureParentControlSet       *picture_control_set_ptr,
    SequenceControlSet            *sequence_control_set_ptr,
    EbInputResolution                 input_resolution)
{
    UNUSED(sequence_control_set_ptr);
    uint8_t  hmeMeLevel =  picture_control_set_ptr->enc_mode; // OMK to be revised after new presets

#if USE_M0_HME_ME_SETTINGS
    hmeMeLevel = 0;
#endif

    // HME/ME default settings
    me_context_ptr->number_hme_search_region_in_width = 2;
    me_context_ptr->number_hme_search_region_in_height = 2;

#if SCREEN_CONTENT_SETTINGS
    uint8_t sc_content_detected = picture_control_set_ptr->sc_content_detected;
#if !DECOUPLE_ALTREF_ME

#if ALTREF_FILTERING_SUPPORT
    if (me_context_ptr->me_alt_ref == EB_TRUE)
        sc_content_detected = 0;
#endif
#endif
#if SCREEN_CONTENT_SETTINGS && !PCS_ME_FIX
    picture_control_set_ptr->enable_hme_level0_flag = enable_hme_level0_flag[sc_content_detected][input_resolution][hmeMeLevel];
    picture_control_set_ptr->enable_hme_level1_flag = enable_hme_level1_flag[sc_content_detected][input_resolution][hmeMeLevel];
    picture_control_set_ptr->enable_hme_level2_flag = enable_hme_level2_flag[sc_content_detected][input_resolution][hmeMeLevel];
#endif
    // HME Level0
    me_context_ptr->hme_level0_total_search_area_width = hme_level0_total_search_area_width[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_total_search_area_height = hme_level0_total_search_area_height[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_width_array[0] = hme_level0_search_area_in_width_array_right[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_width_array[1] = hme_level0_search_area_in_width_array_left[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_height_array[0] = hme_level0_search_area_in_height_array_top[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_height_array[1] = hme_level0_search_area_in_height_array_bottom[sc_content_detected][input_resolution][hmeMeLevel];
    // HME Level1
    me_context_ptr->hme_level1_search_area_in_width_array[0] = hme_level1_search_area_in_width_array_right[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_width_array[1] = hme_level1_search_area_in_width_array_left[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_height_array[0] = hme_level1_search_area_in_height_array_top[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_height_array[1] = hme_level1_search_area_in_height_array_bottom[sc_content_detected][input_resolution][hmeMeLevel];
    // HME Level2
    me_context_ptr->hme_level2_search_area_in_width_array[0] = hme_level2_search_area_in_width_array_right[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_width_array[1] = hme_level2_search_area_in_width_array_left[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_height_array[0] = hme_level2_search_area_in_height_array_top[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_height_array[1] = hme_level2_search_area_in_height_array_bottom[sc_content_detected][input_resolution][hmeMeLevel];

    // ME
    me_context_ptr->search_area_width = search_area_width[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->search_area_height = search_area_height[sc_content_detected][input_resolution][hmeMeLevel];

#if REDUCE_ME_SEARCH_AREA
    assert(me_context_ptr->search_area_width  <= MAX_SEARCH_AREA_WIDTH  && "increase MAX_SEARCH_AREA_WIDTH" );
    assert(me_context_ptr->search_area_height <= MAX_SEARCH_AREA_HEIGHT && "increase MAX_SEARCH_AREA_HEIGHT");
#endif

#else
    // HME Level0
    me_context_ptr->hme_level0_total_search_area_width = hme_level0_total_search_area_width[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_total_search_area_height = hme_level0_total_search_area_height[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_width_array[0] = hme_level0_search_area_in_width_array_right[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_width_array[1] = hme_level0_search_area_in_width_array_left[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_height_array[0] = hme_level0_search_area_in_height_array_top[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_height_array[1] = hme_level0_search_area_in_height_array_bottom[input_resolution][hmeMeLevel];
    // HME Level1
    me_context_ptr->hme_level1_search_area_in_width_array[0] = hme_level1_search_area_in_width_array_right[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_width_array[1] = hme_level1_search_area_in_width_array_left[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_height_array[0] = hme_level1_search_area_in_height_array_top[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_height_array[1] = hme_level1_search_area_in_height_array_bottom[input_resolution][hmeMeLevel];
    // HME Level2
    me_context_ptr->hme_level2_search_area_in_width_array[0] = hme_level2_search_area_in_width_array_right[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_width_array[1] = hme_level2_search_area_in_width_array_left[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_height_array[0] = hme_level2_search_area_in_height_array_top[input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_height_array[1] = hme_level2_search_area_in_height_array_bottom[input_resolution][hmeMeLevel];

    // ME
    me_context_ptr->search_area_width = search_area_width[input_resolution][hmeMeLevel];
    me_context_ptr->search_area_height = search_area_height[input_resolution][hmeMeLevel];
#endif

    me_context_ptr->update_hme_search_center_flag = 1;

    if (input_resolution <= INPUT_SIZE_576p_RANGE_OR_LOWER)
        me_context_ptr->update_hme_search_center_flag = 0;

    return EB_NULL;
};
/******************************************************
* Derive ME Settings for OQ
  Input   : encoder mode and tune
  Output  : ME Kernel signal(s)
******************************************************/
EbErrorType signal_derivation_me_kernel_oq(
    SequenceControlSet        *sequence_control_set_ptr,
    PictureParentControlSet   *picture_control_set_ptr,
    MotionEstimationContext_t   *context_ptr) {
    EbErrorType return_error = EB_ErrorNone;

    // Set ME/HME search regions
    if (sequence_control_set_ptr->static_config.use_default_me_hme)
        set_me_hme_params_oq(
            context_ptr->me_context_ptr,
            picture_control_set_ptr,
            sequence_control_set_ptr,
            sequence_control_set_ptr->input_resolution);

    else
        set_me_hme_params_from_config(
            sequence_control_set_ptr,
            context_ptr->me_context_ptr);
#if SCREEN_CONTENT_SETTINGS
        if (picture_control_set_ptr->sc_content_detected)
#if NEW_M0_SC
            context_ptr->me_context_ptr->fractional_search_method = SUB_SAD_SEARCH;
#else
            if (picture_control_set_ptr->enc_mode <= ENC_M1)
                context_ptr->me_context_ptr->fractional_search_method = SSD_SEARCH ;
            else
                context_ptr->me_context_ptr->fractional_search_method = SUB_SAD_SEARCH;
#endif
        else
#endif
        if (picture_control_set_ptr->enc_mode <= ENC_M6)
        context_ptr->me_context_ptr->fractional_search_method = SSD_SEARCH ;
#if M9_FRAC_ME_SEARCH_METHOD
        else if (picture_control_set_ptr->enc_mode <= ENC_M8)
            context_ptr->me_context_ptr->fractionalSearchMethod = FULL_SAD_SEARCH;
        else
            context_ptr->me_context_ptr->fractionalSearchMethod = SUB_SAD_SEARCH;
#else
        else
        context_ptr->me_context_ptr->fractional_search_method = FULL_SAD_SEARCH;
#endif
#if SCREEN_CONTENT_SETTINGS
        if (picture_control_set_ptr->sc_content_detected)
#if NEW_M0_SC
            context_ptr->me_context_ptr->fractional_search64x64 = EB_FALSE;
#else
            if (picture_control_set_ptr->enc_mode <= ENC_M1)
                context_ptr->me_context_ptr->fractional_search64x64 = EB_TRUE;
            else
                context_ptr->me_context_ptr->fractional_search64x64 = EB_FALSE;
#endif
        else
            context_ptr->me_context_ptr->fractional_search64x64 = EB_TRUE;

#endif
#if M9_FRAC_ME_SEARCH_64x64
    //if (picture_control_set_ptr->sc_content_detected)
    //    context_ptr->fractional_search64x64 = EB_TRUE;
    //else
    {
        if (picture_control_set_ptr->enc_mode <= ENC_M8)
            context_ptr->me_context_ptr->fractional_search64x64 = EB_TRUE;
        else
            context_ptr->me_context_ptr->fractional_search64x64 = EB_FALSE;
    }
#endif

#if DECOUPLE_ALTREF_ME
    // Set HME flags
    context_ptr->me_context_ptr->enable_hme_flag = picture_control_set_ptr->enable_hme_flag;
    context_ptr->me_context_ptr->enable_hme_level0_flag = picture_control_set_ptr->enable_hme_level0_flag;
    context_ptr->me_context_ptr->enable_hme_level1_flag = picture_control_set_ptr->enable_hme_level1_flag;
    context_ptr->me_context_ptr->enable_hme_level2_flag = picture_control_set_ptr->enable_hme_level2_flag;

    // Set the default settings of subpel
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M1)
            context_ptr->me_context_ptr->use_subpel_flag = 1;
        else
            context_ptr->me_context_ptr->use_subpel_flag = 0;
    else
        context_ptr->me_context_ptr->use_subpel_flag = 1;
    if (MR_MODE) {
        context_ptr->me_context_ptr->half_pel_mode =
            EX_HP_MODE;
        context_ptr->me_context_ptr->quarter_pel_mode =
            EX_QP_MODE;
    }
#if M1_HALF_QUARTER_PEL
    else if (0) {
#else
    else if (picture_control_set_ptr->enc_mode ==
        ENC_M0) {
#endif
        context_ptr->me_context_ptr->half_pel_mode =
            EX_HP_MODE;
        context_ptr->me_context_ptr->quarter_pel_mode =
            REFINMENT_QP_MODE;
    }
    else {
        context_ptr->me_context_ptr->half_pel_mode =
            REFINMENT_HP_MODE;
        context_ptr->me_context_ptr->quarter_pel_mode =
            REFINMENT_QP_MODE;
    }
#endif

#if M9_SUBPEL_SELECTION
    // Set fractional search model
    // 0: search all blocks
    // 1: selective based on Full-Search SAD & MV.
    // 2: off
#if DECOUPLE_ALTREF_ME
    if (context_ptr->me_context_ptr->use_subpel_flag == 1) {
#else
    if (picture_control_set_ptr->use_subpel_flag == 1) {
#endif
#if NEW_PRESETS
        if (picture_control_set_ptr->enc_mode <= ENC_M6)
            context_ptr->me_context_ptr->fractional_search_model = 0;
        else
            context_ptr->me_context_ptr->fractional_search_model = 1;
#else
        if (picture_control_set_ptr->enc_mode <= ENC_M8)
            context_ptr->me_context_ptr->fractional_search_model = 0;
        else
            context_ptr->me_context_ptr->fractional_search_model = 1;
#endif
    }
    else
        context_ptr->me_context_ptr->fractional_search_model = 2;
#endif

#if USE_SAD_HME
    // HME Search Method
#if NEW_PRESETS
#if SCREEN_CONTENT_SETTINGS
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M6)
            context_ptr->me_context_ptr->hme_search_method = FULL_SAD_SEARCH;
        else
            context_ptr->me_context_ptr->hme_search_method = SUB_SAD_SEARCH;
    else
#endif
    context_ptr->me_context_ptr->hme_search_method = FULL_SAD_SEARCH;
#else
#if MOD_M0
    context_ptr->me_context_ptr->hme_search_method = SUB_SAD_SEARCH;
#else
    context_ptr->me_context_ptr->hme_search_method = (picture_control_set_ptr->enc_mode == ENC_M0) ?
        FULL_SAD_SEARCH :
        SUB_SAD_SEARCH;
#endif
#endif
#else
    context_ptr->me_context_ptr->hme_search_method = SUB_SAD_SEARCH;
#endif
#if USE_SAD_ME
    // ME Search Method
#if NEW_PRESETS
#if SCREEN_CONTENT_SETTINGS
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M3)
            context_ptr->me_context_ptr->me_search_method = FULL_SAD_SEARCH;
        else
            context_ptr->me_context_ptr->me_search_method = SUB_SAD_SEARCH;
    else
#endif
#if M2_ME_SEARCH_METHOD
        context_ptr->me_context_ptr->me_search_method = SUB_SAD_SEARCH;
#else
    context_ptr->me_context_ptr->me_search_method = (picture_control_set_ptr->enc_mode <= ENC_M1) ?
        FULL_SAD_SEARCH :
        SUB_SAD_SEARCH;
#endif
#else
#if MOD_M0
    context_ptr->me_context_ptr->me_search_method = SUB_SAD_SEARCH;
#else
    context_ptr->me_context_ptr->me_search_method = (picture_control_set_ptr->enc_mode == ENC_M0) ?
        FULL_SAD_SEARCH :
        SUB_SAD_SEARCH;
#endif
#endif
#else
    context_ptr->me_context_ptr->me_search_method = SUB_SAD_SEARCH  ;
#endif
    return return_error;
};

#if DECOUPLE_ALTREF_ME
/************************************************
 * Set ME/HME Params for Altref Temporal Filtering
 ************************************************/
void* tf_set_me_hme_params_oq(
    MeContext               *me_context_ptr,
    PictureParentControlSet *picture_control_set_ptr,
    SequenceControlSet      *sequence_control_set_ptr,
    EbInputResolution        input_resolution)
{
    UNUSED(sequence_control_set_ptr);
    uint8_t  hmeMeLevel = picture_control_set_ptr->enc_mode; // OMK to be revised after new presets

#if M0_SETTINGS
    hmeMeLevel = 0;
#endif

    // HME/ME default settings
    me_context_ptr->number_hme_search_region_in_width = 2;
    me_context_ptr->number_hme_search_region_in_height = 2;

    uint8_t sc_content_detected = picture_control_set_ptr->sc_content_detected;

    // HME Level0
    me_context_ptr->hme_level0_total_search_area_width = tf_hme_level0_total_search_area_width[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_total_search_area_height = tf_hme_level0_total_search_area_height[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_width_array[0] = tf_hme_level0_search_area_in_width_array_right[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_width_array[1] = tf_hme_level0_search_area_in_width_array_left[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_height_array[0] = tf_hme_level0_search_area_in_height_array_top[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level0_search_area_in_height_array[1] = tf_hme_level0_search_area_in_height_array_bottom[sc_content_detected][input_resolution][hmeMeLevel];
    // HME Level1
    me_context_ptr->hme_level1_search_area_in_width_array[0] = tf_hme_level1_search_area_in_width_array_right[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_width_array[1] = tf_hme_level1_search_area_in_width_array_left[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_height_array[0] = tf_hme_level1_search_area_in_height_array_top[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level1_search_area_in_height_array[1] = tf_hme_level1_search_area_in_height_array_bottom[sc_content_detected][input_resolution][hmeMeLevel];
    // HME Level2
    me_context_ptr->hme_level2_search_area_in_width_array[0] = tf_hme_level2_search_area_in_width_array_right[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_width_array[1] = tf_hme_level2_search_area_in_width_array_left[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_height_array[0] = tf_hme_level2_search_area_in_height_array_top[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->hme_level2_search_area_in_height_array[1] = tf_hme_level2_search_area_in_height_array_bottom[sc_content_detected][input_resolution][hmeMeLevel];

    // ME
    me_context_ptr->search_area_width = tf_search_area_width[sc_content_detected][input_resolution][hmeMeLevel];
    me_context_ptr->search_area_height = tf_search_area_height[sc_content_detected][input_resolution][hmeMeLevel];
//
//#if ALTREF_TEMPORAL_FILTERING
//    me_context_ptr->search_area_width = 56;
//    me_context_ptr->search_area_height = 56;
//#endif

    assert(me_context_ptr->search_area_width <= MAX_SEARCH_AREA_WIDTH && "increase MAX_SEARCH_AREA_WIDTH");
    assert(me_context_ptr->search_area_height <= MAX_SEARCH_AREA_HEIGHT && "increase MAX_SEARCH_AREA_HEIGHT");

    me_context_ptr->update_hme_search_center_flag = 1;

    if (input_resolution <= INPUT_SIZE_576p_RANGE_OR_LOWER)
        me_context_ptr->update_hme_search_center_flag = 0;

    return EB_NULL;
};

/******************************************************
* Derive ME Settings for OQ for Altref Temporal Filtering
  Input   : encoder mode and tune
  Output  : ME Kernel signal(s)
******************************************************/
EbErrorType tf_signal_derivation_me_kernel_oq(
    SequenceControlSet        *sequence_control_set_ptr,
    PictureParentControlSet   *picture_control_set_ptr,
    MotionEstimationContext_t *context_ptr) {
    EbErrorType return_error = EB_ErrorNone;

    // Set ME/HME search regions
    tf_set_me_hme_params_oq(
        context_ptr->me_context_ptr,
        picture_control_set_ptr,
        sequence_control_set_ptr,
        sequence_control_set_ptr->input_resolution);

    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M1)
            context_ptr->me_context_ptr->fractional_search_method = SSD_SEARCH;
        else
            context_ptr->me_context_ptr->fractional_search_method = SUB_SAD_SEARCH;
    else
        if (picture_control_set_ptr->enc_mode <= ENC_M6)
            context_ptr->me_context_ptr->fractional_search_method = SSD_SEARCH;
        else
            context_ptr->me_context_ptr->fractional_search_method = FULL_SAD_SEARCH;
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M1)
            context_ptr->me_context_ptr->fractional_search64x64 = EB_TRUE;
        else
            context_ptr->me_context_ptr->fractional_search64x64 = EB_FALSE;
    else
        context_ptr->me_context_ptr->fractional_search64x64 = EB_TRUE;

#if DECOUPLE_ALTREF_ME
    // Set HME flags
    context_ptr->me_context_ptr->enable_hme_flag = picture_control_set_ptr->tf_enable_hme_flag;
    context_ptr->me_context_ptr->enable_hme_level0_flag = picture_control_set_ptr->tf_enable_hme_level0_flag;
    context_ptr->me_context_ptr->enable_hme_level1_flag = picture_control_set_ptr->tf_enable_hme_level1_flag;
    context_ptr->me_context_ptr->enable_hme_level2_flag = picture_control_set_ptr->tf_enable_hme_level2_flag;

    // Set the default settings of subpel
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M1)
            context_ptr->me_context_ptr->use_subpel_flag = 1;
        else
            context_ptr->me_context_ptr->use_subpel_flag = 0;
    else
        context_ptr->me_context_ptr->use_subpel_flag = 1;


#if ALTREF_SHUT_EX_REFINEMENT
    context_ptr->me_context_ptr->half_pel_mode =
        REFINMENT_HP_MODE;
    context_ptr->me_context_ptr->quarter_pel_mode =
        REFINMENT_QP_MODE;
#else
    if (MR_MODE) {
        context_ptr->me_context_ptr->half_pel_mode =
            EX_HP_MODE;
        context_ptr->me_context_ptr->quarter_pel_mode =
            EX_QP_MODE;
    }
#if M1_HALF_QUARTER_PEL
    else if (0) {
#else
    else if (picture_control_set_ptr->enc_mode ==
        ENC_M0) {
#endif
        context_ptr->me_context_ptr->half_pel_mode =
            EX_HP_MODE;
        context_ptr->me_context_ptr->quarter_pel_mode =
            REFINMENT_QP_MODE;
    }
    else {
        context_ptr->me_context_ptr->half_pel_mode =
            REFINMENT_HP_MODE;
        context_ptr->me_context_ptr->quarter_pel_mode =
            REFINMENT_QP_MODE;
    }
#endif


#if ALTREF_ENABLE_EX_REFINEMENT
    context_ptr->me_context_ptr->half_pel_mode =
        EX_HP_MODE;
    context_ptr->me_context_ptr->quarter_pel_mode =
        REFINMENT_QP_MODE;
#endif

#endif

    // Set fractional search model
    // 0: search all blocks
    // 1: selective based on Full-Search SAD & MV.
    // 2: off
#if DECOUPLE_ALTREF_ME
    if (context_ptr->me_context_ptr->use_subpel_flag == 1) {
#else
    if (picture_control_set_ptr->use_subpel_flag == 1) {
#endif
        if (picture_control_set_ptr->enc_mode <= ENC_M6)
            context_ptr->me_context_ptr->fractional_search_model = 0;
        else
            context_ptr->me_context_ptr->fractional_search_model = 1;
    }
    else
        context_ptr->me_context_ptr->fractional_search_model = 2;

    // HME Search Method
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M6)
            context_ptr->me_context_ptr->hme_search_method = FULL_SAD_SEARCH;
        else
            context_ptr->me_context_ptr->hme_search_method = SUB_SAD_SEARCH;
    else
        context_ptr->me_context_ptr->hme_search_method = FULL_SAD_SEARCH;
    // ME Search Method
    if (picture_control_set_ptr->sc_content_detected)
        if (picture_control_set_ptr->enc_mode <= ENC_M3)
            context_ptr->me_context_ptr->me_search_method = FULL_SAD_SEARCH;
        else
            context_ptr->me_context_ptr->me_search_method = SUB_SAD_SEARCH;
    else
#if M2_ME_SEARCH_METHOD
        context_ptr->me_context_ptr->me_search_method =
        SUB_SAD_SEARCH;
#else
        context_ptr->me_context_ptr->me_search_method = (picture_control_set_ptr->enc_mode <= ENC_M1) ?
        FULL_SAD_SEARCH :
        SUB_SAD_SEARCH;
#endif
    return return_error;
    };
#endif

/************************************************
 * Motion Analysis Context Constructor
 ************************************************/
#if MEMORY_FOOTPRINT_OPT_ME_MV
EbErrorType motion_estimation_context_ctor(
    MotionEstimationContext_t   **context_dbl_ptr,
    EbFifo                       *picture_decision_results_input_fifo_ptr,
    EbFifo                       *motion_estimation_results_output_fifo_ptr,
#if REDUCE_ME_SEARCH_AREA
    uint16_t                      max_input_luma_width,
    uint16_t                      max_input_luma_height,
#endif
    uint8_t                       nsq_present,
    uint8_t                       mrp_mode) {
#else
EbErrorType motion_estimation_context_ctor(
    MotionEstimationContext_t   **context_dbl_ptr,
    EbFifo                     *picture_decision_results_input_fifo_ptr,
    EbFifo                     *motion_estimation_results_output_fifo_ptr) {
#endif

    EbErrorType return_error = EB_ErrorNone;
    MotionEstimationContext_t *context_ptr;
    EB_MALLOC(MotionEstimationContext_t*, context_ptr, sizeof(MotionEstimationContext_t), EB_N_PTR);

    *context_dbl_ptr = context_ptr;

    context_ptr->picture_decision_results_input_fifo_ptr = picture_decision_results_input_fifo_ptr;
    context_ptr->motion_estimation_results_output_fifo_ptr = motion_estimation_results_output_fifo_ptr;
    return_error = intra_open_loop_reference_samples_ctor(&context_ptr->intra_ref_ptr);
    if (return_error == EB_ErrorInsufficientResources)
        return EB_ErrorInsufficientResources;
#if MEMORY_FOOTPRINT_OPT_ME_MV
    return_error = me_context_ctor(
        &(context_ptr->me_context_ptr),
#if REDUCE_ME_SEARCH_AREA
        max_input_luma_width,
        max_input_luma_height,
#endif
        nsq_present,
        mrp_mode);
#else
    return_error = me_context_ctor(&(context_ptr->me_context_ptr));
#endif
    if (return_error == EB_ErrorInsufficientResources)
        return EB_ErrorInsufficientResources;
    return EB_ErrorNone;
}

/***************************************************************************************************
* ZZ Decimated SAD Computation
***************************************************************************************************/
#if !ADAPTIVE_QP_SCALING
int non_moving_th_shift[4] = { 4, 2, 0, 0 };
#endif
EbErrorType ComputeDecimatedZzSad(
    MotionEstimationContext_t   *context_ptr,
    SequenceControlSet        *sequence_control_set_ptr,
    PictureParentControlSet   *picture_control_set_ptr,
    EbPictureBufferDesc       *sixteenth_decimated_picture_ptr,
    uint32_t                         xLcuStartIndex,
    uint32_t                         xLcuEndIndex,
    uint32_t                         yLcuStartIndex,
    uint32_t                         yLcuEndIndex) {
    EbErrorType return_error = EB_ErrorNone;

    EbAsm asm_type = sequence_control_set_ptr->encode_context_ptr->asm_type;

    PictureParentControlSet    *previous_picture_control_set_wrapper_ptr = ((PictureParentControlSet*)picture_control_set_ptr->previous_picture_control_set_wrapper_ptr->object_ptr);
    EbPictureBufferDesc        *previousInputPictureFull = previous_picture_control_set_wrapper_ptr->enhanced_picture_ptr;

    uint32_t sb_index;

    uint32_t sb_width;
    uint32_t sb_height;

    uint32_t decimatedLcuWidth;
    uint32_t decimatedLcuHeight;

    uint32_t sb_origin_x;
    uint32_t sb_origin_y;

    uint32_t blkDisplacementDecimated;
    uint32_t blkDisplacementFull;

    uint32_t decimatedLcuCollocatedSad;

    uint32_t x_lcu_index;
    uint32_t y_lcu_index;

    for (y_lcu_index = yLcuStartIndex; y_lcu_index < yLcuEndIndex; ++y_lcu_index) {
        for (x_lcu_index = xLcuStartIndex; x_lcu_index < xLcuEndIndex; ++x_lcu_index) {
            sb_index = x_lcu_index + y_lcu_index * sequence_control_set_ptr->picture_width_in_sb;
            SbParams *sb_params = &sequence_control_set_ptr->sb_params_array[sb_index];

            sb_width = sb_params->width;
            sb_height = sb_params->height;

            sb_origin_x = sb_params->origin_x;
            sb_origin_y = sb_params->origin_y;

            sb_width = sb_params->width;
            sb_height = sb_params->height;

            decimatedLcuWidth = sb_width >> 2;
            decimatedLcuHeight = sb_height >> 2;

            decimatedLcuCollocatedSad = 0;

            if (sb_params->is_complete_sb)
            {
                blkDisplacementDecimated = (sixteenth_decimated_picture_ptr->origin_y + (sb_origin_y >> 2)) * sixteenth_decimated_picture_ptr->stride_y + sixteenth_decimated_picture_ptr->origin_x + (sb_origin_x >> 2);
                blkDisplacementFull = (previousInputPictureFull->origin_y + sb_origin_y)* previousInputPictureFull->stride_y + (previousInputPictureFull->origin_x + sb_origin_x);

                // 1/16 collocated SB decimation
                decimation_2d(
                    &previousInputPictureFull->buffer_y[blkDisplacementFull],
                    previousInputPictureFull->stride_y,
                    BLOCK_SIZE_64,
                    BLOCK_SIZE_64,
                    context_ptr->me_context_ptr->sixteenth_sb_buffer,
                    context_ptr->me_context_ptr->sixteenth_sb_buffer_stride,
                    4);

                if (asm_type >= ASM_NON_AVX2 && asm_type < ASM_TYPE_TOTAL)
                    // ZZ SAD between 1/16 current & 1/16 collocated
                    decimatedLcuCollocatedSad = nxm_sad_kernel_func_ptr_array[asm_type][2](
                        &(sixteenth_decimated_picture_ptr->buffer_y[blkDisplacementDecimated]),
                        sixteenth_decimated_picture_ptr->stride_y,
                        context_ptr->me_context_ptr->sixteenth_sb_buffer,
                        context_ptr->me_context_ptr->sixteenth_sb_buffer_stride,
                        16, 16);
#if !MEMORY_FOOTPRINT_OPT
                // Background Enhancement Algorithm
                // Classification is important to:
                // 1. Avoid improving moving objects.
                // 2. Do not modulate when all the picture is background
                // 3. Do give different importance to different regions
                if (decimatedLcuCollocatedSad < BEA_CLASS_0_0_DEC_TH)
                    previous_picture_control_set_wrapper_ptr->zz_cost_array[sb_index] = BEA_CLASS_0_ZZ_COST;
                else if (decimatedLcuCollocatedSad < BEA_CLASS_0_DEC_TH)
                    previous_picture_control_set_wrapper_ptr->zz_cost_array[sb_index] = BEA_CLASS_0_1_ZZ_COST;
                else if (decimatedLcuCollocatedSad < BEA_CLASS_1_DEC_TH)
                    previous_picture_control_set_wrapper_ptr->zz_cost_array[sb_index] = BEA_CLASS_1_ZZ_COST;
                else if (decimatedLcuCollocatedSad < BEA_CLASS_2_DEC_TH)
                    previous_picture_control_set_wrapper_ptr->zz_cost_array[sb_index] = BEA_CLASS_2_ZZ_COST;
                else
                    previous_picture_control_set_wrapper_ptr->zz_cost_array[sb_index] = BEA_CLASS_3_ZZ_COST;
#endif
            }
            else {
#if !MEMORY_FOOTPRINT_OPT
                previous_picture_control_set_wrapper_ptr->zz_cost_array[sb_index] = INVALID_ZZ_COST;
#endif
                decimatedLcuCollocatedSad = (uint32_t)~0;
            }
#if ADAPTIVE_QP_SCALING
            // Keep track of non moving LCUs for QP modulation
            if (decimatedLcuCollocatedSad < ((decimatedLcuWidth * decimatedLcuHeight) * 2))
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_0_ZZ_COST;
            else if (decimatedLcuCollocatedSad < ((decimatedLcuWidth * decimatedLcuHeight) * 4))
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_1_ZZ_COST;
            else if (decimatedLcuCollocatedSad < ((decimatedLcuWidth * decimatedLcuHeight) * 8))
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_2_ZZ_COST;
            else
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_3_ZZ_COST;
#else
            // Keep track of non moving LCUs for QP modulation
            if (decimatedLcuCollocatedSad < ((decimatedLcuWidth * decimatedLcuHeight) * 2) >> non_moving_th_shift[sequence_control_set_ptr->input_resolution])
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_0_ZZ_COST;
            else if (decimatedLcuCollocatedSad < ((decimatedLcuWidth * decimatedLcuHeight) * 4) >> non_moving_th_shift[sequence_control_set_ptr->input_resolution])
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_1_ZZ_COST;
            else if (decimatedLcuCollocatedSad < ((decimatedLcuWidth * decimatedLcuHeight) * 8) >> non_moving_th_shift[sequence_control_set_ptr->input_resolution])
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_2_ZZ_COST;
            else
                previous_picture_control_set_wrapper_ptr->non_moving_index_array[sb_index] = BEA_CLASS_3_ZZ_COST;
#endif
        }
    }

    return return_error;
}

/************************************************
 * Motion Analysis Kernel
 * The Motion Analysis performs  Motion Estimation
 * This process has access to the current input picture as well as
 * the input pictures, which the current picture references according
 * to the prediction structure pattern.  The Motion Analysis process is multithreaded,
 * so pictures can be processed out of order as long as all inputs are available.
 ************************************************/
void* motion_estimation_kernel(void *input_ptr)
{
    MotionEstimationContext_t   *context_ptr = (MotionEstimationContext_t*)input_ptr;

    PictureParentControlSet   *picture_control_set_ptr;
    SequenceControlSet        *sequence_control_set_ptr;

    EbObjectWrapper           *inputResultsWrapperPtr;
    PictureDecisionResults    *inputResultsPtr;

    EbObjectWrapper           *outputResultsWrapperPtr;
    MotionEstimationResults   *outputResultsPtr;

    EbPictureBufferDesc       *input_picture_ptr;

    EbPictureBufferDesc       *input_padded_picture_ptr;

    uint32_t                       bufferIndex;

    uint32_t                       sb_index;
    uint32_t                       x_lcu_index;
    uint32_t                       y_lcu_index;
    uint32_t                       picture_width_in_sb;
    uint32_t                       picture_height_in_sb;
    uint32_t                       sb_origin_x;
    uint32_t                       sb_origin_y;
    uint32_t                       sb_width;
    uint32_t                       sb_height;
    uint32_t                       lcuRow;

    EbPaReferenceObject       *paReferenceObject;
#if DOWN_SAMPLING_FILTERING
    EbPictureBufferDesc       *quarter_picture_ptr;
    EbPictureBufferDesc       *sixteenth_picture_ptr;
#else
    EbPictureBufferDesc       *quarter_decimated_picture_ptr;
    EbPictureBufferDesc       *sixteenth_decimated_picture_ptr;
#endif

    // Segments
    uint32_t                      segment_index;
    uint32_t                      xSegmentIndex;
    uint32_t                      ySegmentIndex;
    uint32_t                      xLcuStartIndex;
    uint32_t                      xLcuEndIndex;
    uint32_t                      yLcuStartIndex;
    uint32_t                      yLcuEndIndex;

    uint32_t                      intra_sad_interval_index;

    EbAsm                      asm_type;
#if !ENABLE_CDF_UPDATE
    MdRateEstimationContext   *md_rate_estimation_array;
#endif
    for (;;) {
        // Get Input Full Object
        eb_get_full_object(
            context_ptr->picture_decision_results_input_fifo_ptr,
            &inputResultsWrapperPtr);

        inputResultsPtr = (PictureDecisionResults*)inputResultsWrapperPtr->object_ptr;
        picture_control_set_ptr = (PictureParentControlSet*)inputResultsPtr->picture_control_set_wrapper_ptr->object_ptr;
        sequence_control_set_ptr = (SequenceControlSet*)picture_control_set_ptr->sequence_control_set_wrapper_ptr->object_ptr;

        paReferenceObject = (EbPaReferenceObject*)picture_control_set_ptr->pa_reference_picture_wrapper_ptr->object_ptr;
#if DOWN_SAMPLING_FILTERING
        // Set 1/4 and 1/16 ME input buffer(s); filtered or decimated
        quarter_picture_ptr = (sequence_control_set_ptr->down_sampling_method_me_search == ME_FILTERED_DOWNSAMPLED) ?
            (EbPictureBufferDesc*)paReferenceObject->quarter_filtered_picture_ptr :
            (EbPictureBufferDesc*)paReferenceObject->quarter_decimated_picture_ptr;

        sixteenth_picture_ptr = (sequence_control_set_ptr->down_sampling_method_me_search == ME_FILTERED_DOWNSAMPLED) ?
            (EbPictureBufferDesc*)paReferenceObject->sixteenth_filtered_picture_ptr :
            (EbPictureBufferDesc*)paReferenceObject->sixteenth_decimated_picture_ptr;
#else
        quarter_decimated_picture_ptr = (EbPictureBufferDesc*)paReferenceObject->quarter_decimated_picture_ptr;
        sixteenth_decimated_picture_ptr = (EbPictureBufferDesc*)paReferenceObject->sixteenth_decimated_picture_ptr;
#endif

        input_padded_picture_ptr = (EbPictureBufferDesc*)paReferenceObject->input_padded_picture_ptr;

        input_picture_ptr = picture_control_set_ptr->enhanced_picture_ptr;

        asm_type = sequence_control_set_ptr->encode_context_ptr->asm_type;
#if !ENABLE_CDF_UPDATE
        // Increment the MD Rate Estimation array pointer to point to the right address based on the QP and slice type
        md_rate_estimation_array = (MdRateEstimationContext*)sequence_control_set_ptr->encode_context_ptr->md_rate_estimation_array;
        md_rate_estimation_array += picture_control_set_ptr->slice_type * TOTAL_NUMBER_OF_QP_VALUES + picture_control_set_ptr->picture_qp;
        // Reset MD rate Estimation table to initial values by copying from md_rate_estimation_array
        EB_MEMCPY(&(context_ptr->me_context_ptr->mvd_bits_array[0]), &(md_rate_estimation_array->mvd_bits[0]), sizeof(EbBitFraction)*NUMBER_OF_MVD_CASES);
        ///context_ptr->me_context_ptr->lambda = lambda_mode_decision_ld_sad_qp_scaling[picture_control_set_ptr->picture_qp];
#endif
#if ALTREF_FILTERING_SUPPORT
        context_ptr->me_context_ptr->me_alt_ref = inputResultsPtr->task_type == 1 ? EB_TRUE : EB_FALSE;
#endif
#if !DECOUPLE_ALTREF_ME
        // ME Kernel Signal(s) derivation
        signal_derivation_me_kernel_oq(
            sequence_control_set_ptr,
            picture_control_set_ptr,
            context_ptr);
#endif
        // Lambda Assignement
        if (sequence_control_set_ptr->static_config.pred_structure == EB_PRED_RANDOM_ACCESS) {
            if (picture_control_set_ptr->temporal_layer_index == 0)
                context_ptr->me_context_ptr->lambda = lambda_mode_decision_ra_sad[picture_control_set_ptr->picture_qp];
            else if (picture_control_set_ptr->temporal_layer_index < 3)
                context_ptr->me_context_ptr->lambda = lambda_mode_decision_ra_sad_qp_scaling_l1[picture_control_set_ptr->picture_qp];
            else
                context_ptr->me_context_ptr->lambda = lambda_mode_decision_ra_sad_qp_scaling_l3[picture_control_set_ptr->picture_qp];
        }
        else {
            if (picture_control_set_ptr->temporal_layer_index == 0)
                context_ptr->me_context_ptr->lambda = lambda_mode_decision_ld_sad[picture_control_set_ptr->picture_qp];
            else
                context_ptr->me_context_ptr->lambda = lambda_mode_decision_ld_sad_qp_scaling[picture_control_set_ptr->picture_qp];
        }
#if ALTREF_FILTERING_SUPPORT
        if (inputResultsPtr->task_type == 0)
        {
#endif

#if DECOUPLE_ALTREF_ME
            // ME Kernel Signal(s) derivation
            signal_derivation_me_kernel_oq(
                sequence_control_set_ptr,
                picture_control_set_ptr,
                context_ptr);
#endif
                        // Segments
            segment_index = inputResultsPtr->segment_index;
            picture_width_in_sb = (sequence_control_set_ptr->seq_header.max_frame_width + sequence_control_set_ptr->sb_sz - 1) / sequence_control_set_ptr->sb_sz;
            picture_height_in_sb = (sequence_control_set_ptr->seq_header.max_frame_height + sequence_control_set_ptr->sb_sz - 1) / sequence_control_set_ptr->sb_sz;
            SEGMENT_CONVERT_IDX_TO_XY(segment_index, xSegmentIndex, ySegmentIndex, picture_control_set_ptr->me_segments_column_count);
            xLcuStartIndex = SEGMENT_START_IDX(xSegmentIndex, picture_width_in_sb, picture_control_set_ptr->me_segments_column_count);
            xLcuEndIndex = SEGMENT_END_IDX(xSegmentIndex, picture_width_in_sb, picture_control_set_ptr->me_segments_column_count);
            yLcuStartIndex = SEGMENT_START_IDX(ySegmentIndex, picture_height_in_sb, picture_control_set_ptr->me_segments_row_count);
            yLcuEndIndex = SEGMENT_END_IDX(ySegmentIndex, picture_height_in_sb, picture_control_set_ptr->me_segments_row_count);
            // *** MOTION ESTIMATION CODE ***
            if (picture_control_set_ptr->slice_type != I_SLICE) {
                // SB Loop
                for (y_lcu_index = yLcuStartIndex; y_lcu_index < yLcuEndIndex; ++y_lcu_index) {
                    for (x_lcu_index = xLcuStartIndex; x_lcu_index < xLcuEndIndex; ++x_lcu_index) {
                        sb_index = (uint16_t)(x_lcu_index + y_lcu_index * picture_width_in_sb);
                        sb_origin_x = x_lcu_index * sequence_control_set_ptr->sb_sz;
                        sb_origin_y = y_lcu_index * sequence_control_set_ptr->sb_sz;

                        sb_width = (sequence_control_set_ptr->seq_header.max_frame_width - sb_origin_x) < BLOCK_SIZE_64 ? sequence_control_set_ptr->seq_header.max_frame_width - sb_origin_x : BLOCK_SIZE_64;
                        sb_height = (sequence_control_set_ptr->seq_header.max_frame_height - sb_origin_y) < BLOCK_SIZE_64 ? sequence_control_set_ptr->seq_header.max_frame_height - sb_origin_y : BLOCK_SIZE_64;

                        // Load the SB from the input to the intermediate SB buffer
                        bufferIndex = (input_picture_ptr->origin_y + sb_origin_y) * input_picture_ptr->stride_y + input_picture_ptr->origin_x + sb_origin_x;

                        context_ptr->me_context_ptr->hme_search_type = HME_RECTANGULAR;

                        for (lcuRow = 0; lcuRow < BLOCK_SIZE_64; lcuRow++) {
                            EB_MEMCPY((&(context_ptr->me_context_ptr->sb_buffer[lcuRow * BLOCK_SIZE_64])), (&(input_picture_ptr->buffer_y[bufferIndex + lcuRow * input_picture_ptr->stride_y])), BLOCK_SIZE_64 * sizeof(uint8_t));
                        }

                        {
                            uint8_t * src_ptr = &input_padded_picture_ptr->buffer_y[bufferIndex];

                            //_MM_HINT_T0     //_MM_HINT_T1    //_MM_HINT_T2//_MM_HINT_NTA
                            uint32_t i;
                            for (i = 0; i < sb_height; i++)
                            {
                                char const* p = (char const*)(src_ptr + i * input_padded_picture_ptr->stride_y);
                                _mm_prefetch(p, _MM_HINT_T2);
                            }
                        }

                        context_ptr->me_context_ptr->sb_src_ptr = &input_padded_picture_ptr->buffer_y[bufferIndex];
                        context_ptr->me_context_ptr->sb_src_stride = input_padded_picture_ptr->stride_y;

#if DOWN_SAMPLING_FILTERING
                        // Load the 1/4 decimated SB from the 1/4 decimated input to the 1/4 intermediate SB buffer
#if DECOUPLE_ALTREF_ME
                        if (context_ptr->me_context_ptr->enable_hme_level1_flag) {
#else
                        if (picture_control_set_ptr->enable_hme_level1_flag) {
#endif
                            bufferIndex = (quarter_picture_ptr->origin_y + (sb_origin_y >> 1)) * quarter_picture_ptr->stride_y + quarter_picture_ptr->origin_x + (sb_origin_x >> 1);

                            for (lcuRow = 0; lcuRow < (sb_height >> 1); lcuRow++) {
                                EB_MEMCPY((&(context_ptr->me_context_ptr->quarter_sb_buffer[lcuRow * context_ptr->me_context_ptr->quarter_sb_buffer_stride])), (&(quarter_picture_ptr->buffer_y[bufferIndex + lcuRow * quarter_picture_ptr->stride_y])), (sb_width >> 1) * sizeof(uint8_t));
                            }
                        }

                        // Load the 1/16 decimated SB from the 1/16 decimated input to the 1/16 intermediate SB buffer
#if DECOUPLE_ALTREF_ME
                        if (context_ptr->me_context_ptr->enable_hme_level0_flag) {
#else
                        if (picture_control_set_ptr->enable_hme_level0_flag) {
#endif
                            bufferIndex = (sixteenth_picture_ptr->origin_y + (sb_origin_y >> 2)) * sixteenth_picture_ptr->stride_y + sixteenth_picture_ptr->origin_x + (sb_origin_x >> 2);

                            {
                                uint8_t  *framePtr = &sixteenth_picture_ptr->buffer_y[bufferIndex];
                                uint8_t  *localPtr = context_ptr->me_context_ptr->sixteenth_sb_buffer;
#if USE_SAD_HMEL0
                                if (context_ptr->me_context_ptr->hme_search_method == FULL_SAD_SEARCH) {
                                    for (lcuRow = 0; lcuRow < (sb_height >> 2); lcuRow += 1) {
                                        EB_MEMCPY(localPtr, framePtr, (sb_width >> 2) * sizeof(uint8_t));
                                        localPtr += 16;
                                        framePtr += sixteenth_picture_ptr->stride_y;
                                    }
                                }
                                else {
                                    for (lcuRow = 0; lcuRow < (sb_height >> 2); lcuRow += 2) {
                                        EB_MEMCPY(localPtr, framePtr, (sb_width >> 2) * sizeof(uint8_t));
                                        localPtr += 16;
                                        framePtr += sixteenth_picture_ptr->stride_y << 1;
                                    }
                                }
#else
                                for (lcuRow = 0; lcuRow < (sb_height >> 2); lcuRow += 2) {
                                    EB_MEMCPY(localPtr, framePtr, (sb_width >> 2) * sizeof(uint8_t));
                                    localPtr += 16;
                                    framePtr += sixteenth_decimated_picture_ptr->stride_y << 1;
                                }
#endif
                            }
                        }
#else
                        // Load the 1/4 decimated SB from the 1/4 decimated input to the 1/4 intermediate SB buffer
                        if (picture_control_set_ptr->enable_hme_level1_flag) {
                            bufferIndex = (quarter_decimated_picture_ptr->origin_y + (sb_origin_y >> 1)) * quarter_decimated_picture_ptr->stride_y + quarter_decimated_picture_ptr->origin_x + (sb_origin_x >> 1);

                            for (lcuRow = 0; lcuRow < (sb_height >> 1); lcuRow++) {
                                EB_MEMCPY((&(context_ptr->me_context_ptr->quarter_sb_buffer[lcuRow * context_ptr->me_context_ptr->quarter_sb_buffer_stride])), (&(quarter_decimated_picture_ptr->buffer_y[bufferIndex + lcuRow * quarter_decimated_picture_ptr->stride_y])), (sb_width >> 1) * sizeof(uint8_t));
                            }
                        }

                        // Load the 1/16 decimated SB from the 1/16 decimated input to the 1/16 intermediate SB buffer
                        if (picture_control_set_ptr->enable_hme_level0_flag) {
                            bufferIndex = (sixteenth_decimated_picture_ptr->origin_y + (sb_origin_y >> 2)) * sixteenth_decimated_picture_ptr->stride_y + sixteenth_decimated_picture_ptr->origin_x + (sb_origin_x >> 2);

                            {
                                uint8_t  *framePtr = &sixteenth_decimated_picture_ptr->buffer_y[bufferIndex];
                                uint8_t  *localPtr = context_ptr->me_context_ptr->sixteenth_sb_buffer;
#if USE_SAD_HMEL0
                                if (context_ptr->me_context_ptr->hme_search_method == FULL_SAD_SEARCH) {
                                    for (lcuRow = 0; lcuRow < (sb_height >> 2); lcuRow += 1) {
                                        EB_MEMCPY(localPtr, framePtr, (sb_width >> 2) * sizeof(uint8_t));
                                        localPtr += 16;
                                        framePtr += sixteenth_decimated_picture_ptr->stride_y;
                                    }
                                }
                                else {
                                    for (lcuRow = 0; lcuRow < (sb_height >> 2); lcuRow += 2) {
                                        EB_MEMCPY(localPtr, framePtr, (sb_width >> 2) * sizeof(uint8_t));
                                        localPtr += 16;
                                        framePtr += sixteenth_decimated_picture_ptr->stride_y << 1;
                                    }
                                }
#else
                                for (lcuRow = 0; lcuRow < (sb_height >> 2); lcuRow += 2) {
                                    EB_MEMCPY(localPtr, framePtr, (sb_width >> 2) * sizeof(uint8_t));
                                    localPtr += 16;
                                    framePtr += sixteenth_decimated_picture_ptr->stride_y << 1;
                                }
#endif
                            }
                        }
#endif

#if ALTREF_FILTERING_SUPPORT
                        context_ptr->me_context_ptr->me_alt_ref = EB_FALSE;
#endif

                        motion_estimate_lcu(
                            picture_control_set_ptr,
                            sb_index,
                            sb_origin_x,
                            sb_origin_y,
                            context_ptr->me_context_ptr,
                            input_picture_ptr);
                    }
                }
            }

#if M9_INTRA && !ADD_MDC_INTRA
        if ( picture_control_set_ptr->intra_pred_mode > 4)
#endif
                // *** OPEN LOOP INTRA CANDIDATE SEARCH CODE ***
            {
                // SB Loop
                for (y_lcu_index = yLcuStartIndex; y_lcu_index < yLcuEndIndex; ++y_lcu_index) {
                    for (x_lcu_index = xLcuStartIndex; x_lcu_index < xLcuEndIndex; ++x_lcu_index) {
                        sb_origin_x = x_lcu_index * sequence_control_set_ptr->sb_sz;
                        sb_origin_y = y_lcu_index * sequence_control_set_ptr->sb_sz;

                        sb_index = (uint16_t)(x_lcu_index + y_lcu_index * picture_width_in_sb);

                        open_loop_intra_search_sb(
                            picture_control_set_ptr,
                            sb_index,
                            context_ptr,
                            input_picture_ptr,
                            asm_type);
                    }
                }
            }

            // ZZ SADs Computation
            // 1 lookahead frame is needed to get valid (0,0) SAD
            if (sequence_control_set_ptr->static_config.look_ahead_distance != 0) {
                // when DG is ON, the ZZ SADs are computed @ the PD process
                {
                    // ZZ SADs Computation using decimated picture
                    if (picture_control_set_ptr->picture_number > 0) {
                        ComputeDecimatedZzSad(
                            context_ptr,
                            sequence_control_set_ptr,
                            picture_control_set_ptr,
#if DOWN_SAMPLING_FILTERING
                            (EbPictureBufferDesc*)paReferenceObject->sixteenth_decimated_picture_ptr, // Hsan: always use decimated for ZZ SAD derivation until studying the trade offs and regenerating the activity threshold
#else
                            sixteenth_decimated_picture_ptr,
#endif
                            xLcuStartIndex,
                            xLcuEndIndex,
                            yLcuStartIndex,
                            yLcuEndIndex);
                    }
                }
            }

            // Calculate the ME Distortion and OIS Historgrams

            eb_block_on_mutex(picture_control_set_ptr->rc_distortion_histogram_mutex);

            if (sequence_control_set_ptr->static_config.rate_control_mode) {
                if (picture_control_set_ptr->slice_type != I_SLICE) {
                    uint16_t sadIntervalIndex;
                    for (y_lcu_index = yLcuStartIndex; y_lcu_index < yLcuEndIndex; ++y_lcu_index) {
                        for (x_lcu_index = xLcuStartIndex; x_lcu_index < xLcuEndIndex; ++x_lcu_index) {
                            sb_origin_x = x_lcu_index * sequence_control_set_ptr->sb_sz;
                            sb_origin_y = y_lcu_index * sequence_control_set_ptr->sb_sz;
                            sb_width = (sequence_control_set_ptr->seq_header.max_frame_width - sb_origin_x) < BLOCK_SIZE_64 ? sequence_control_set_ptr->seq_header.max_frame_width - sb_origin_x : BLOCK_SIZE_64;
                            sb_height = (sequence_control_set_ptr->seq_header.max_frame_height - sb_origin_y) < BLOCK_SIZE_64 ? sequence_control_set_ptr->seq_header.max_frame_height - sb_origin_y : BLOCK_SIZE_64;

                            sb_index = (uint16_t)(x_lcu_index + y_lcu_index * picture_width_in_sb);
                            picture_control_set_ptr->inter_sad_interval_index[sb_index] = 0;
                            picture_control_set_ptr->intra_sad_interval_index[sb_index] = 0;

                            if (sb_width == BLOCK_SIZE_64 && sb_height == BLOCK_SIZE_64) {
                                sadIntervalIndex = (uint16_t)(picture_control_set_ptr->rc_me_distortion[sb_index] >> (12 - SAD_PRECISION_INTERVAL));//change 12 to 2*log2(64)

                                // printf("%d\n", sadIntervalIndex);

                                sadIntervalIndex = (uint16_t)(sadIntervalIndex >> 2);
                                if (sadIntervalIndex > (NUMBER_OF_SAD_INTERVALS >> 1) - 1) {
                                    uint16_t sadIntervalIndexTemp = sadIntervalIndex - ((NUMBER_OF_SAD_INTERVALS >> 1) - 1);

                                    sadIntervalIndex = ((NUMBER_OF_SAD_INTERVALS >> 1) - 1) + (sadIntervalIndexTemp >> 3);
                                }
                                if (sadIntervalIndex >= NUMBER_OF_SAD_INTERVALS - 1)
                                    sadIntervalIndex = NUMBER_OF_SAD_INTERVALS - 1;

                                picture_control_set_ptr->inter_sad_interval_index[sb_index] = sadIntervalIndex;

                                picture_control_set_ptr->me_distortion_histogram[sadIntervalIndex] ++;
#if RC

                                intra_sad_interval_index = picture_control_set_ptr->variance[sb_index][ME_TIER_ZERO_PU_64x64] >> 4;
#else
                                uint32_t                       bestOisCuIndex = 0;

                                //DOUBLE CHECK THIS PIECE OF CODE
                                bestOisCuIndex = picture_control_set_ptr->ois_sb_results[sb_index]->best_distortion_index[0];
                            intra_sad_interval_index = (uint32_t) ( picture_control_set_ptr->ois_sb_results[sb_index]->ois_candidate_array[0][bestOisCuIndex].distortion  >> (12 - SAD_PRECISION_INTERVAL));//change 12 to 2*log2(64) ;
#endif
                                intra_sad_interval_index = (uint16_t)(intra_sad_interval_index >> 2);
                                if (intra_sad_interval_index > (NUMBER_OF_SAD_INTERVALS >> 1) - 1) {
                                    uint32_t sadIntervalIndexTemp = intra_sad_interval_index - ((NUMBER_OF_SAD_INTERVALS >> 1) - 1);

                                    intra_sad_interval_index = ((NUMBER_OF_SAD_INTERVALS >> 1) - 1) + (sadIntervalIndexTemp >> 3);
                                }
                                if (intra_sad_interval_index >= NUMBER_OF_SAD_INTERVALS - 1)
                                    intra_sad_interval_index = NUMBER_OF_SAD_INTERVALS - 1;

                                picture_control_set_ptr->intra_sad_interval_index[sb_index] = intra_sad_interval_index;

                                picture_control_set_ptr->ois_distortion_histogram[intra_sad_interval_index] ++;

                                ++picture_control_set_ptr->full_sb_count;
                            }
                        }
                    }
                }
                else {
#if !RC
                    uint32_t                       bestOisCuIndex = 0;
#endif
                    for (y_lcu_index = yLcuStartIndex; y_lcu_index < yLcuEndIndex; ++y_lcu_index) {
                        for (x_lcu_index = xLcuStartIndex; x_lcu_index < xLcuEndIndex; ++x_lcu_index) {
                            sb_origin_x = x_lcu_index * sequence_control_set_ptr->sb_sz;
                            sb_origin_y = y_lcu_index * sequence_control_set_ptr->sb_sz;
                            sb_width = (sequence_control_set_ptr->seq_header.max_frame_width - sb_origin_x) < BLOCK_SIZE_64 ? sequence_control_set_ptr->seq_header.max_frame_width - sb_origin_x : BLOCK_SIZE_64;
                            sb_height = (sequence_control_set_ptr->seq_header.max_frame_height - sb_origin_y) < BLOCK_SIZE_64 ? sequence_control_set_ptr->seq_header.max_frame_height - sb_origin_y : BLOCK_SIZE_64;

                            sb_index = (uint16_t)(x_lcu_index + y_lcu_index * picture_width_in_sb);

                            picture_control_set_ptr->inter_sad_interval_index[sb_index] = 0;
                            picture_control_set_ptr->intra_sad_interval_index[sb_index] = 0;

                            if (sb_width == BLOCK_SIZE_64 && sb_height == BLOCK_SIZE_64) {
#if RC

                                intra_sad_interval_index = picture_control_set_ptr->variance[sb_index][ME_TIER_ZERO_PU_64x64] >> 4;
#else
                                uint32_t                       bestOisCuIndex = 0;

                                bestOisCuIndex = picture_control_set_ptr->ois_sb_results[sb_index]->best_distortion_index[0];
                            intra_sad_interval_index = (uint32_t) ( picture_control_set_ptr->ois_sb_results[sb_index]->ois_candidate_array[0][bestOisCuIndex].distortion  >> (12 - SAD_PRECISION_INTERVAL));//change 12 to 2*log2(64) ;
#endif
                                intra_sad_interval_index = (uint16_t)(intra_sad_interval_index >> 2);
                                if (intra_sad_interval_index > (NUMBER_OF_SAD_INTERVALS >> 1) - 1) {
                                    uint32_t sadIntervalIndexTemp = intra_sad_interval_index - ((NUMBER_OF_SAD_INTERVALS >> 1) - 1);

                                    intra_sad_interval_index = ((NUMBER_OF_SAD_INTERVALS >> 1) - 1) + (sadIntervalIndexTemp >> 3);
                                }
                                if (intra_sad_interval_index >= NUMBER_OF_SAD_INTERVALS - 1)
                                    intra_sad_interval_index = NUMBER_OF_SAD_INTERVALS - 1;

                                picture_control_set_ptr->intra_sad_interval_index[sb_index] = intra_sad_interval_index;

                                picture_control_set_ptr->ois_distortion_histogram[intra_sad_interval_index] ++;

                                ++picture_control_set_ptr->full_sb_count;
                            }
                        }
                    }
                }
            }

            eb_release_mutex(picture_control_set_ptr->rc_distortion_histogram_mutex);

            // Get Empty Results Object
            eb_get_empty_object(
                context_ptr->motion_estimation_results_output_fifo_ptr,
                &outputResultsWrapperPtr);

            outputResultsPtr = (MotionEstimationResults*)outputResultsWrapperPtr->object_ptr;
            outputResultsPtr->picture_control_set_wrapper_ptr = inputResultsPtr->picture_control_set_wrapper_ptr;
            outputResultsPtr->segment_index = segment_index;

            // Release the Input Results
            eb_release_object(inputResultsWrapperPtr);

            // Post the Full Results Object
            eb_post_full_object(outputResultsWrapperPtr);

#if ALTREF_FILTERING_SUPPORT
        }
        else {
#if DECOUPLE_ALTREF_ME
        // ME Kernel Signal(s) derivation
        tf_signal_derivation_me_kernel_oq(
            sequence_control_set_ptr,
            picture_control_set_ptr,
            context_ptr);
#endif
            // temporal filtering start
            context_ptr->me_context_ptr->me_alt_ref = EB_TRUE;
            init_temporal_filtering(picture_control_set_ptr->temp_filt_pcs_list, picture_control_set_ptr, context_ptr, inputResultsPtr->segment_index);

             // Release the Input Results
             eb_release_object(inputResultsWrapperPtr);
        }
#endif
    }

    return EB_NULL;
}
