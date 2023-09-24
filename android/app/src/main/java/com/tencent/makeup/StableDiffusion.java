// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package com.tencent.makeup;

import android.content.res.AssetManager;
import android.graphics.Bitmap;

public class StableDiffusion
{
    public native boolean Init(AssetManager mgr);

    public native boolean gen(Bitmap show_bitmap, int seed);

    public native boolean setPoint(Bitmap show_bitmap, int x, int y);

    public native boolean clean(Bitmap show_bitmap);

    public native boolean drag(Bitmap show_bitmap, int step);

    static {
        System.loadLibrary("makeup");
    }
}
