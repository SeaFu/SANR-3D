from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG
from .spconv_backbone import VirConvL8x, VirConv8x
__all__ = {
    'VirConvL8x': VirConvL8x,
    'VirConv8x': VirConv8x,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
}
