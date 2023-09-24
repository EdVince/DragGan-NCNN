// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
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
import android.annotation.SuppressLint;
import android.content.Context;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity
{
    private ImageView imageView;
    private EditText seedText;

    private Bitmap showBitmap;

    private StableDiffusion sd = new StableDiffusion();
    /** Called when the activity is first created. */
    @SuppressLint("MissingInflatedId")
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        imageView = (ImageView) findViewById(R.id.resView);
        seedText = (EditText) findViewById(R.id.seed);
        showBitmap = Bitmap.createBitmap(512,512,Bitmap.Config.ARGB_8888);

        boolean ret_init = sd.Init(getAssets());

        Button buttonGenImage = (Button) findViewById(R.id.gen);
        buttonGenImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                int seed = Integer.parseInt(seedText.getText().toString());
                sd.gen(showBitmap, seed);
                final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888,true);
                imageView.post(new Runnable() {
                    public void run() {
                        imageView.setImageBitmap(styledImage);
                        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                    }
                });
            }
        });

        imageView.setOnTouchListener(new View.OnTouchListener() {
            boolean shouldProcessTouch = true;
            long lastTouchTime = 0;
            final long TOUCH_DELAY = 500;  // 设置触摸延迟时间为500毫秒

            @Override
            public boolean onTouch(View v, MotionEvent event) {
                long now = System.currentTimeMillis();

                if (now - lastTouchTime < TOUCH_DELAY) {
                    shouldProcessTouch = false;  // 如果距离上次触摸时间小于延迟时间，则不处理触摸事件
                } else {
                    shouldProcessTouch = true;
                    lastTouchTime = now;
                }

                if(!shouldProcessTouch){
                    return true;
                }

                int x = (int) event.getX();
                int y = (int) event.getY();

                int[] location = new int[2];
                v.getLocationOnScreen(location);

                int relativeX = x - location[0];
                int relativeY = y - location[1];

                relativeX = 512 * relativeX / v.getWidth();
                relativeY = 512 * relativeY / v.getWidth();

                // 处理点击事件，使用 x 和 y
                Log.e("MainActivity", "("+x+","+y+")  ("+relativeX+","+relativeY+")");
                sd.setPoint(showBitmap,relativeX,relativeY);
                final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888,true);
                imageView.post(new Runnable() {
                    public void run() {
                        imageView.setImageBitmap(styledImage);
                        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                    }
                });

                return true;
            }
        });

        Button buttonClean = (Button) findViewById(R.id.clean);
        buttonClean.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                sd.clean(showBitmap);
                final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888,true);
                imageView.post(new Runnable() {
                    public void run() {
                        imageView.setImageBitmap(styledImage);
                        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                    }
                });
            }
        });

        Button buttonDrag = (Button) findViewById(R.id.drag);
        buttonDrag.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                new Thread(new Runnable() {
                    @Override
                    public void run() {
                        for (int it = 0; it < 200; it++) {
                            boolean bk = sd.drag(showBitmap, it);
                            final Bitmap styledImage = showBitmap.copy(Bitmap.Config.ARGB_8888, true);
                            final int finalIt = it;
                            runOnUiThread(new Runnable() {
                                public void run() {
                                    Toast.makeText(getApplicationContext(), "Step:"+ finalIt, Toast.LENGTH_SHORT).show();
                                    imageView.setImageBitmap(styledImage);
                                    getWindow().clearFlags(WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE);
                                }
                            });
                            if(bk){
                                runOnUiThread(new Runnable() {
                                    public void run() {
                                        Toast.makeText(getApplicationContext(), "Finish!", Toast.LENGTH_SHORT).show();
                                    }
                                });
                                break;
                            }
                        }
                    }
                }).start();
            }
        });

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);
        switch (requestCode) {
            case 1:
                if (resultCode == RESULT_OK && null != data) {
                    Uri selectedImage = data.getData();
                    try {
                        final Bitmap tmp = decodeUri(selectedImage);
                        showBitmap = Bitmap.createScaledBitmap(tmp,512,512,false);
                        imageView.setImageBitmap(showBitmap);
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException(e);
                    }
                }
                break;
            default:
                break;
        }
    }

    private void copy(Context myContext, String ASSETS_NAME, String savePath, String saveName) {
        String filename = savePath + "/" + saveName;
        File dir = new File(savePath);
        if (!dir.exists())
            dir.mkdir();
        try {
            if (!(new File(filename)).exists()) {
                InputStream is = myContext.getResources().getAssets().open(ASSETS_NAME);
                FileOutputStream fos = new FileOutputStream(filename);
                byte[] buffer = new byte[7168];
                int count = 0;
                while ((count = is.read(buffer)) > 0) {
                    fos.write(buffer, 0, count);
                }
                fos.close();
                is.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 512;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (true) {
            if (width_tmp / 2 < REQUIRED_SIZE
                    || height_tmp / 2 < REQUIRED_SIZE) {
                break;
            }
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
}
