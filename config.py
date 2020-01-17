data_dict = {
             0:{
                      "inshape":(1, 1, 64, 128, 128),
                      "kershape":(24, 1, 1, 5, 5),
                      "outshape":(1, 24, 64, 64, 64),
                      "stride":(1, 2, 2),
                      "padding":(0, 2, 2),
                      "dilation":1,
                      "groups":1,
                      "bias": None
                     },
#Slower than cudnn, I do not use it    
#              1:{
#                       "inshape":(1, 24, 64, 128, 128),
#                       "kershape":(12, 24, 1, 1, 1),
#                       "outshape":(1, 12, 64, 128, 128),
#                       "stride":(1, 1, 1),
#                       "padding":(0, 0, 0),
#                       "dilation":1,
#                        "groups":1,
#                        "bias": None
#                       },
    
            2:{
                      "inshape":(1, 32, 64, 128, 128),
                      "kershape":(12, 32,1, 1, 1),
                      "outshape":(1, 12, 64, 128, 128),
                      "stride":(1, 1, 1),
                      "padding":(0, 0, 0),
                      "dilation":1,
                       "groups":1,
                       "bias": None
                      },
            3:{
                      "inshape":(1, 24, 64, 64, 64),
                      "kershape":(24, 24, 3, 3, 3),
                      "outshape":(1, 24, 64, 64, 64),
                      "stride":(1, 1, 1),
                      "padding":(1, 1, 1),
                      "dilation":1,
                       "groups":1,
                       "bias": None
                      },  
            4:{
                      "inshape":(1, 64, 32, 32, 32),
                      "kershape":(64,64,3,3,3),
                      "outshape":(1, 64, 32, 32, 32),
                      "stride":(1, 1, 1),
                      "padding":(1, 1, 1),
                      "dilation":1,
                       "groups":1,
                       "bias": None
                      },
            5:{
                      "inshape":(1, 192, 16, 16, 16),
                      "kershape":(192,192,3,3,3),
                      "outshape":(1, 192, 16, 16, 16),
                      "stride":(1, 1, 1),
                      "padding":(1, 1, 1),
                      "dilation":1,
                       "groups":1,
                       "bias": None
                      },
            6:{
                      "inshape":(1, 192, 8, 8, 8),
                      "kershape":(192,192, 3, 3, 3),
                      "outshape":(1, 192, 8, 8, 8),
                      "stride":(1, 1, 1),
                      "padding":(1, 1, 1),
                      "dilation":1,
                       "groups":1,
                       "bias": None
                      },
            }