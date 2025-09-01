# unet_action.py
# PlotNeuralNet spec for a U-Net with action conditioning at the bottleneck
# Mirrors the provided PyTorch code structure.

from pycore.tikzeng import *
from pycore.blocks  import *

# ---- Config you might tweak ----
IMG_H, IMG_W = 256, 256          # diagram scaling only
MODE        = "interpolate"       # "interpolate" (6ch) or "predict" (3ch)
IN_CH       = 6 if MODE == "interpolate" else 3
ACT_DIM     = 128                 # action_mlp output (after 2nd ReLU) in your code
ENC_CH      = [64, 128, 256, 512] # encoder out channels
BOT_CH      = 1024
DEC_CH      = [512, 256, 128, 64] # decoder out channels
OUT_FRAMES  = 1                   # set >1 if predicting multiple frames; head outputs 3*frames
OUT_CH      = 3*OUT_FRAMES

# Utility “double conv” label
def dbl(label_c):
    return f"Conv3×3→ReLU→Conv3×3→ReLU\n{label_c} ch"

arch = [
    to_head( '..' ),
    to_cor(),
    to_begin(),

    # ----- INPUT -----
    to_Conv( "input", s_filer=str((IMG_W, IMG_H)), n_filer=IN_CH,
             offset="(0,0,0)", to="(0,0,0)", height=28, depth=28, width=2, caption=f"Input ({IN_CH} ch)"),

    # ----- Encoder -----
    # enc1
    to_Conv( "enc1", s_filer=str((IMG_W, IMG_H)), n_filer=ENC_CH[0],
             offset="(2,0,0)", to="(input-east)", height=28, depth=28, width=4,
             caption=dbl(ENC_CH[0]) ),
    to_Pool( "pool1", offset="(0,0,0)", to="(enc1-east)", width=1, height=22, depth=22),

    # enc2
    to_Conv( "enc2", s_filer=str((IMG_W//2, IMG_H//2)), n_filer=ENC_CH[1],
             offset="(2,0,0)", to="(pool1-east)", height=22, depth=22, width=6,
             caption=dbl(ENC_CH[1]) ),
    to_Pool( "pool2", offset="(0,0,0)", to="(enc2-east)", width=1, height=16, depth=16),

    # enc3
    to_Conv( "enc3", s_filer=str((IMG_W//4, IMG_H//4)), n_filer=ENC_CH[2],
             offset="(2,0,0)", to="(pool2-east)", height=16, depth=16, width=8,
             caption=dbl(ENC_CH[2]) ),
    to_Pool( "pool3", offset="(0,0,0)", to="(enc3-east)", width=1, height=12, depth=12),

    # enc4
    to_Conv( "enc4", s_filer=str((IMG_W//8, IMG_H//8)), n_filer=ENC_CH[3],
             offset="(2,0,0)", to="(pool3-east)", height=12, depth=12, width=10,
             caption=dbl(ENC_CH[3]) ),
    to_Pool( "pool4", offset="(0,0,0)", to="(enc4-east)", width=1, height=8, depth=8),

    # ----- Bottleneck -----
    to_Conv( "bottleneck", s_filer=str((IMG_W//16, IMG_H//16)), n_filer=BOT_CH,
             offset="(2,0,0)", to="(pool4-east)", height=8, depth=8, width=12,
             caption=dbl(BOT_CH) ),

    # ----- Action MLP branch (side) -----
    # Draw a vertical MLP stack to the south of the bottleneck
    to_Node( "act_in",  offset="(0,-4,0)", to="(bottleneck-south)",  caption=f"Actions ({ACT_DIM//2}→{ACT_DIM})" ),
    to_Node( "mlp1",    offset="(1,0,0)",  to="(act_in-east)", caption="Linear 64\nReLU" ),
    to_Node( "mlp2",    offset="(1,0,0)",  to="(mlp1-east)",   caption="Linear 128\nReLU" ),
    to_Connection( "act_in-east", "mlp1-west" ),
    to_Connection( "mlp1-east", "mlp2-west" ),

    # Fusion projection (per-channel projection back to F = H×W)
    to_Node( "fuseproj", offset="(1.5,2.0,0)", to="(bottleneck-south)",
             caption="FuseProj\nLinear(F+D→F)" ),

    # Connect MLP→FuseProj→Bottleneck (conceptual arrows)
    to_Connection( "mlp2-north", "fuseproj-south" ),
    to_Connection( "fuseproj-north", "bottleneck-south" ),

    # ----- Decoder (+ skip concats) -----
    # up1: 1024→512, concat with enc4 → dec1(512)
    to_DeConv( "up1", s_filer=str((IMG_W//8, IMG_H//8)), n_filer=DEC_CH[0],
               offset="(2.5,0,0)", to="(bottleneck-east)", height=12, depth=12, width=10,
               caption="UpConv 2×" ),
    to_Concat( "cat1", offset="(0,0,0)", to="(up1-east)", radius=2.0, opacity=0.6 ),
    to_Conv( "dec1", s_filer=str((IMG_W//8, IMG_H//8)), n_filer=DEC_CH[0],
             offset="(1.5,0,0)", to="(cat1-east)", height=12, depth=12, width=10,
             caption=dbl(DEC_CH[0]) ),

    # up2: 512→256, concat with enc3 → dec2(256)
    to_DeConv( "up2", s_filer=str((IMG_W//4, IMG_H//4)), n_filer=DEC_CH[1],
               offset="(2.5,0,0)", to="(dec1-east)", height=16, depth=16, width=8,
               caption="UpConv 2×" ),
    to_Concat( "cat2", offset="(0,0,0)", to="(up2-east)", radius=2.0, opacity=0.6 ),
    to_Conv( "dec2", s_filer=str((IMG_W//4, IMG_H//4)), n_filer=DEC_CH[1],
             offset="(1.5,0,0)", to="(cat2-east)", height=16, depth=16, width=8,
             caption=dbl(DEC_CH[1]) ),

    # up3: 256→128, concat with enc2 → dec3(128)
    to_DeConv( "up3", s_filer=str((IMG_W//2, IMG_H//2)), n_filer=DEC_CH[2],
               offset="(2.5,0,0)", to="(dec2-east)", height=22, depth=22, width=6,
               caption="UpConv 2×" ),
    to_Concat( "cat3", offset="(0,0,0)", to="(up3-east)", radius=2.0, opacity=0.6 ),
    to_Conv( "dec3", s_filer=str((IMG_W//2, IMG_H//2)), n_filer=DEC_CH[2],
             offset="(1.5,0,0)", to="(cat3-east)", height=22, depth=22, width=6,
             caption=dbl(DEC_CH[2]) ),

    # up4: 128→64, concat with enc1 → dec4(64)
    to_DeConv( "up4", s_filer=str((IMG_W, IMG_H)), n_filer=DEC_CH[3],
               offset="(2.5,0,0)", to="(dec3-east)", height=28, depth=28, width=4,
               caption="UpConv 2×" ),
    to_Concat( "cat4", offset="(0,0,0)", to="(up4-east)", radius=2.0, opacity=0.6 ),
    to_Conv( "dec4", s_filer=str((IMG_W, IMG_H)), n_filer=DEC_CH[3],
             offset="(1.5,0,0)", to="(cat4-east)", height=28, depth=28, width=4,
             caption=dbl(DEC_CH[3]) ),

    # ----- Output head -----
    to_Conv( "out", s_filer=str((IMG_W, IMG_H)), n_filer=OUT_CH,
             offset="(2,0,0)", to="(dec4-east)", height=28, depth=28, width=2,
             caption=f"1×1 Conv → Sigmoid\n{OUT_CH} ch" ),

    # ----- Skip connections (encoder → concat nodes) -----
    to_connection( "enc4-east", "cat1-west" ),
    to_connection( "enc3-east", "cat2-west" ),
    to_connection( "enc2-east", "cat3-west" ),
    to_connection( "enc1-east", "cat4-west" ),

    # ----- Main flow connections -----
    to_connection( "input-east", "enc1-west" ),
    to_connection( "enc1-east", "pool1-west" ),
    to_connection( "pool1-east", "enc2-west" ),
    to_connection( "enc2-east", "pool2-west" ),
    to_connection( "pool2-east", "enc3-west" ),
    to_connection( "enc3-east", "pool3-west" ),
    to_connection( "pool3-east", "enc4-west" ),
    to_connection( "enc4-east", "pool4-west" ),
    to_connection( "pool4-east", "bottleneck-west" ),
    to_connection( "bottleneck-east", "up1-west" ),
    to_connection( "up1-east", "cat1-west" ),
    to_connection( "cat1-east", "dec1-west" ),
    to_connection( "dec1-east", "up2-west" ),
    to_connection( "up2-east", "cat2-west" ),
    to_connection( "cat2-east", "dec2-west" ),
    to_connection( "dec2-east", "up3-west" ),
    to_connection( "up3-east", "cat3-west" ),
    to_connection( "cat3-east", "dec3-west" ),
    to_connection( "dec3-east", "up4-west" ),
    to_connection( "up4-east", "cat4-west" ),
    to_connection( "cat4-east", "dec4-west" ),
    to_connection( "dec4-east", "out-west" ),

    to_end()
]

def main():
    name = 'unet_action'
    to_generate(arch, name + '.tex')

if __name__ == '__main__':
    main()
