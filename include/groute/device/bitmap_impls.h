//
// Created by gengl on 18-11-14.
//

#ifndef SEP_GRAPH_BITMAP_IMPLS_H
#define SEP_GRAPH_BITMAP_IMPLS_H

#include <groute/device/compressed_bitmap.cuh>
#include <groute/device/array_bitmap.h>

#ifdef ARRAY_BITMAP
    typedef sepgraph::ArrayBitmap Bitmap;
    typedef sepgraph::dev::ArrayBitmap BitmapDeviceObject;
#else
    typedef sepgraph::CompressedBitmap Bitmap;
    typedef sepgraph::dev::CompressedBitmap BitmapDeviceObject;
#endif

#endif //SEP_GRAPH_BITMAP_IMPLS_H
