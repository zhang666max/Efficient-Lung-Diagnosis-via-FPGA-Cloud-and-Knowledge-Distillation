#include "fpga_cnn.h"
#include "weights.h"
#include <hls_stream.h>

// =========================================================================
// 辅助函数：ReLU 激活
// =========================================================================
data_t relu(data_t x) {
    #pragma HLS INLINE
    return (x > 0) ? x : (data_t)0;
}

// =========================================================================
// 专用卷积模板 A：Factor = 3 (针对 Layer 1，输入通道=3)
// =========================================================================
template<int IN_CH, int OUT_CH, int IN_DIM>
void conv_layer_factor3(hls::stream<data_t> &in, hls::stream<data_t> &out, 
                        const weight_t W[OUT_CH][IN_CH][3][3], const weight_t B[OUT_CH]) {
    #pragma HLS INLINE off
    
    // 行缓存与滑动窗口
    static data_t line_buf[2][IN_DIM][IN_CH];
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram
    // 维度3(通道)完全切分，配合 factor=3 实现全并行
    #pragma HLS ARRAY_PARTITION variable=line_buf dim=3 complete 

    data_t window[3][3][IN_CH];
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0

    // 遍历图像
    for (int r = 0; r < IN_DIM; r++) {
        for (int c = 0; c < IN_DIM; c++) {
            #pragma HLS PIPELINE II=1

            // 1. 读取新数据并更新窗口
            for (int ch = 0; ch < IN_CH; ch++) {
                // 读取输入流
                data_t val_in = in.read();
                
                // 移位操作更新 Window
                window[0][0][ch] = window[0][1][ch]; window[0][1][ch] = window[0][2][ch];
                window[1][0][ch] = window[1][1][ch]; window[1][1][ch] = window[1][2][ch];
                window[2][0][ch] = window[2][1][ch]; window[2][1][ch] = window[2][2][ch];

                // 从 LineBuffer 填充窗口
                data_t lb_val0 = line_buf[0][c][ch];
                data_t lb_val1 = line_buf[1][c][ch];
                
                window[0][2][ch] = lb_val0;
                window[1][2][ch] = lb_val1;
                window[2][2][ch] = val_in;

                // 更新 LineBuffer
                line_buf[0][c][ch] = lb_val1;
                line_buf[1][c][ch] = val_in;
            }

            // 2. 卷积计算 (并行度核心)
            for (int o = 0; o < OUT_CH; o++) {
                acc_t acc = B[o]; 
                
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        for (int ich = 0; ich < IN_CH; ich++) {
                            // *** 汇报重点：Factor=3 完全展开 ***
                            #pragma HLS UNROLL factor=3
                            acc += window[kh][kw][ich] * W[o][ich][kh][kw];
                        }
                    }
                }
                out.write(relu((data_t)acc));
            }
        }
    }
}

// =========================================================================
// 专用卷积模板 B：Factor = 4 (针对 Layer 2 & 3，计算瓶颈层)
// =========================================================================
template<int IN_CH, int OUT_CH, int IN_DIM>
void conv_layer_factor4(hls::stream<data_t> &in, hls::stream<data_t> &out, 
                        const weight_t W[OUT_CH][IN_CH][3][3], const weight_t B[OUT_CH]) {
    #pragma HLS INLINE off
    
    static data_t line_buf[2][IN_DIM][IN_CH];
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram
    // 切分因子设为 4，匹配 Unroll Factor
    #pragma HLS ARRAY_PARTITION variable=line_buf dim=3 factor=4 cyclic 

    data_t window[3][3][IN_CH];
    #pragma HLS ARRAY_PARTITION variable=window dim=3 factor=4 cyclic

    for (int r = 0; r < IN_DIM; r++) {
        for (int c = 0; c < IN_DIM; c++) {
            #pragma HLS PIPELINE II=1

            // 更新窗口逻辑
            for (int ch = 0; ch < IN_CH; ch++) {
                data_t val = in.read();
                
                win_shift:
                window[0][0][ch] = window[0][1][ch]; window[0][1][ch] = window[0][2][ch];
                window[1][0][ch] = window[1][1][ch]; window[1][1][ch] = window[1][2][ch];
                window[2][0][ch] = window[2][1][ch]; window[2][1][ch] = window[2][2][ch];
                
                window[0][2][ch] = line_buf[0][c][ch];
                window[1][2][ch] = line_buf[1][c][ch];
                window[2][2][ch] = val;
                
                line_buf[0][c][ch] = line_buf[1][c][ch];
                line_buf[1][c][ch] = val;
            }

            // 卷积计算
            for (int o = 0; o < OUT_CH; o++) {
                acc_t acc = B[o];
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        for (int ich = 0; ich < IN_CH; ich++) {
                            // *** 汇报重点：Factor=4 强力并行 ***
                            #pragma HLS UNROLL factor=4
                            acc += window[kh][kw][ich] * W[o][ich][kh][kw];
                        }
                    }
                }
                out.write(relu((data_t)acc));
            }
        }
    }
}

// =========================================================================
// 专用卷积模板 C：Factor = 2 (针对 Layer 4，节省资源层)
// =========================================================================
template<int IN_CH, int OUT_CH, int IN_DIM>
void conv_layer_factor2(hls::stream<data_t> &in, hls::stream<data_t> &out, 
                        const weight_t W[OUT_CH][IN_CH][3][3], const weight_t B[OUT_CH]) {
    #pragma HLS INLINE off
    
    static data_t line_buf[2][IN_DIM][IN_CH];
    #pragma HLS BIND_STORAGE variable=line_buf type=ram_2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=line_buf dim=3 factor=2 cyclic 

    data_t window[3][3][IN_CH];
    #pragma HLS ARRAY_PARTITION variable=window dim=3 factor=2 cyclic

    for (int r = 0; r < IN_DIM; r++) {
        for (int c = 0; c < IN_DIM; c++) {
            #pragma HLS PIPELINE II=1

            for (int ch = 0; ch < IN_CH; ch++) {
                data_t val = in.read();
                
                window[0][0][ch] = window[0][1][ch]; window[0][1][ch] = window[0][2][ch];
                window[1][0][ch] = window[1][1][ch]; window[1][1][ch] = window[1][2][ch];
                window[2][0][ch] = window[2][1][ch]; window[2][1][ch] = window[2][2][ch];
                
                window[0][2][ch] = line_buf[0][c][ch];
                window[1][2][ch] = line_buf[1][c][ch];
                window[2][2][ch] = val;
                
                line_buf[0][c][ch] = line_buf[1][c][ch];
                line_buf[1][c][ch] = val;
            }

            for (int o = 0; o < OUT_CH; o++) {
                acc_t acc = B[o];
                for (int kh = 0; kh < 3; kh++) {
                    for (int kw = 0; kw < 3; kw++) {
                        for (int ich = 0; ich < IN_CH; ich++) {
                            // *** 汇报重点：Factor=2 降低资源占用 ***
                            #pragma HLS UNROLL factor=2
                            acc += window[kh][kw][ich] * W[o][ich][kh][kw];
                        }
                    }
                }
                out.write(relu((data_t)acc));
            }
        }
    }
}

// =========================================================================
// 最大池化层 (2x2 Max Pooling)
// =========================================================================
template<int CH, int DIM>
void max_pool(hls::stream<data_t> &in, hls::stream<data_t> &out) {
    #pragma HLS INLINE off
    static data_t buf[DIM][CH]; 
    #pragma HLS ARRAY_PARTITION variable=buf dim=2 factor=4 cyclic 

    for (int r = 0; r < DIM; r++) {
        for (int c = 0; c < DIM; c++) {
            #pragma HLS PIPELINE II=1
            for (int ch = 0; ch < CH; ch++) {
                data_t val = in.read();
                if (r % 2 == 0 && c % 2 == 0) {
                    buf[c/2][ch] = val; 
                } else if (r % 2 == 0 && c % 2 == 1) {
                    if (val > buf[c/2][ch]) buf[c/2][ch] = val; 
                } else if (r % 2 == 1 && c % 2 == 0) {
                    if (val > buf[c/2][ch]) buf[c/2][ch] = val; 
                } else {
                    data_t max = (val > buf[c/2][ch]) ? val : buf[c/2][ch]; 
                    out.write(max);
                }
            }
        }
    }
}

// =========================================================================
// 全连接层 (FC Layer) - 修复变量引用问题
// =========================================================================
void fc_layer(hls::stream<data_t> &in, hls::stream<data_t> &out) {
    #pragma HLS INLINE off
    
    acc_t acc[FC_OUT];
    #pragma HLS ARRAY_PARTITION variable=acc complete
    
    // 初始化偏置
    // 直接引用全局变量 fc_bias (在 weights.h 中定义)
    for (int o = 0; o < FC_OUT; o++) {
        #pragma HLS UNROLL
        acc[o] = (acc_t)fc_bias[o];
    }

    // 全连接计算
    for (int i = 0; i < FC_IN; i++) {
        #pragma HLS PIPELINE II=1
        data_t val = in.read();
        
        for (int o = 0; o < FC_OUT; o++) {
            #pragma HLS UNROLL
            // 直接引用全局变量 fc_weights (1D 数组形式)
            // 确保 weights.h 中定义的是: static weight_t fc_weights[FC_IN * FC_OUT] ...
            acc_t w = (acc_t)fc_weights[o * FC_IN + i];
            
            // 使用 DSP 加速
            acc_t p = (acc_t)val * w;
            #pragma HLS BIND_OP variable=p op=mul impl=dsp latency=3
            acc[o] += p;
        }
    }
    
    // 输出结果
    for (int o = 0; o < FC_OUT; o++) {
        out.write((data_t)acc[o]);
    }
}

// =========================================================================
// 顶层函数 (Top Level)
// =========================================================================
void fpga_cnn(bus_t *in_data, bus_t *out_data) {
    #pragma HLS INTERFACE m_axi port=in_data offset=slave bundle=gmem depth=12288
    #pragma HLS INTERFACE m_axi port=out_data offset=slave bundle=gmem depth=1
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    static hls::stream<data_t> s_in, s_c1, s_p1, s_c2, s_p2, s_c3, s_p3, s_c4, s_p4, s_fc;
    #pragma HLS STREAM variable=s_in depth=128
    #pragma HLS STREAM variable=s_c1 depth=64
    #pragma HLS STREAM variable=s_p1 depth=64
    
    // DATAFLOW: 任务级流水线
    #pragma HLS DATAFLOW

    // 1. 数据解包 (Bus -> Stream)
    for (int i = 0; i < (IMG_H * IMG_W * IMG_CH) / 4; i++) {
        #pragma HLS PIPELINE II=1
        bus_t pack = in_data[i];
        for (int k = 0; k < 4; k++) {
            data_t pixel = pack.range(16*(k+1)-1, 16*k);
            s_in.write(pixel);
        }
    }

    // 2. Layer 1: Conv(3->16) + Pool(128->64) [Factor=3]
    // 注意: 调用时直接使用 weights.h 中的全局变量名
    conv_layer_factor3<3, 16, 128>(s_in, s_c1, conv1_weights, conv1_bias);
    max_pool<16, 128>(s_c1, s_p1);

    // 3. Layer 2: Conv(16->32) + Pool(64->32) [Factor=4]
    conv_layer_factor4<16, 32, 64>(s_p1, s_c2, conv2_weights, conv2_bias);
    max_pool<32, 64>(s_c2, s_p2);

    // 4. Layer 3: Conv(32->64) + Pool(32->16) [Factor=4]
    conv_layer_factor4<32, 64, 32>(s_p2, s_c3, conv3_weights, conv3_bias);
    max_pool<64, 32>(s_c3, s_p3);

    // 5. Layer 4: Conv(64->128) + Pool(16->8) [Factor=2]
    conv_layer_factor2<64, 128, 16>(s_p3, s_c4, conv4_weights, conv4_bias);
    max_pool<128, 16>(s_c4, s_p4); 

    // 6. FC Layer
    fc_layer(s_p4, s_fc);

    // 7. 结果打包 (Stream -> Bus)
    bus_t res_pack = 0;
    for (int k = 0; k < 4; k++) {
        data_t score = s_fc.read();
        res_pack.range(16*(k+1)-1, 16*k) = score(15, 0);
    }
    out_data[0] = res_pack;
}