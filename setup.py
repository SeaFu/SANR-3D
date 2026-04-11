import os
import subprocess

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_git_commit_number():
    if not os.path.exists(".git"):
        return "0000000"
    cmd_out = subprocess.run(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE)
    return cmd_out.stdout.decode("utf-8")[:7]


def make_cuda_ext(name, module, sources):
    return CUDAExtension(
        name=f"{module}.{name}",
        sources=[os.path.join(*module.split("."), src) for src in sources],
    )


def write_version_to_file(version, target_file):
    with open(target_file, "w") as f:
        print(f'__version__ = "{version}"', file=f)


def env_true(key: str, default: str = "0") -> bool:
    return os.environ.get(key, default).strip().lower() in ("1", "true", "yes", "y")


if __name__ == "__main__":
    version = "0.3.0+%s" % get_git_commit_number()
    write_version_to_file(version, "pcdet/version.py")

    # === Build switches ===
    # Torch2.x does NOT ship THC headers, so these legacy ops will fail unless patched:
    # - votr_ops
    # - pointnet2_stack / pointnet2_batch (your repo uses THC/THC.h)
    BUILD_VOTR = env_true("ENABLE_VOTR", "0")
    BUILD_POINTNET2 = env_true("ENABLE_POINTNET2", "0")

    if BUILD_VOTR:
        print("[WARN] ENABLE_VOTR=1 requested. This repo's votr_ops uses THC headers and "
              "will likely fail on torch2.x unless you patch the C++ sources.")
    else:
        print("[INFO] Skip building votr_ops_cuda (default).")

    if BUILD_POINTNET2:
        print("[WARN] ENABLE_POINTNET2=1 requested. This repo's pointnet2 ops use THC headers and "
              "will likely fail on torch2.x unless you patch the C++ sources.")
    else:
        print("[INFO] Skip building pointnet2_*_cuda (default).")

    ext_modules = []

    # ---- Optional: VOTR ops (OFF by default) ----
    if BUILD_VOTR:
        ext_modules.append(
            make_cuda_ext(
                name="votr_ops_cuda",
                module="pcdet.ops.votr_ops",
                sources=[
                    "src/votr_api.cpp",
                    "src/build_mapping.cpp",
                    "src/build_mapping_gpu.cu",
                    "src/build_attention_indices.cpp",
                    "src/build_attention_indices_gpu.cu",
                    "src/group_features.cpp",
                    "src/group_features_gpu.cu",
                ],
            )
        )

    # ---- Core ops (KITTI commonly needs these) ----
    ext_modules += [
        make_cuda_ext(
            name="iou3d_nms_cuda",
            module="pcdet.ops.iou3d_nms",
            sources=[
                "src/iou3d_cpu.cpp",
                "src/iou3d_nms_api.cpp",
                "src/iou3d_nms.cpp",
                "src/iou3d_nms_kernel.cu",
            ],
        ),
        make_cuda_ext(
            name="roiaware_pool3d_cuda",
            module="pcdet.ops.roiaware_pool3d",
            sources=[
                "src/roiaware_pool3d.cpp",
                "src/roiaware_pool3d_kernel.cu",
            ],
        ),
        make_cuda_ext(
            name="roipoint_pool3d_cuda",
            module="pcdet.ops.roipoint_pool3d",
            sources=[
                "src/roipoint_pool3d.cpp",
                "src/roipoint_pool3d_kernel.cu",
            ],
        ),
    ]

    # ---- Optional: PointNet2 ops (OFF by default; uses THC in this repo) ----
    if BUILD_POINTNET2:
        ext_modules += [
            make_cuda_ext(
                name="pointnet2_stack_cuda",
                module="pcdet.ops.pointnet2.pointnet2_stack",
                sources=[
                    "src/pointnet2_api.cpp",
                    "src/ball_query.cpp",
                    "src/ball_query_gpu.cu",
                    "src/group_points.cpp",
                    "src/group_points_gpu.cu",
                    "src/sampling.cpp",
                    "src/sampling_gpu.cu",
                    "src/interpolate.cpp",
                    "src/interpolate_gpu.cu",
                    "src/voxel_query.cpp",
                    "src/voxel_query_gpu.cu",
                    "src/ball_query_deform.cpp",
                    "src/ball_query_deform_gpu.cu",
                    "src/vector_pool.cpp",
                    "src/vector_pool_gpu.cu",
                ],
            ),
            make_cuda_ext(
                name="pointnet2_batch_cuda",
                module="pcdet.ops.pointnet2.pointnet2_batch",
                sources=[
                    "src/pointnet2_api.cpp",
                    "src/ball_query.cpp",
                    "src/ball_query_gpu.cu",
                    "src/group_points.cpp",
                    "src/group_points_gpu.cu",
                    "src/interpolate.cpp",
                    "src/interpolate_gpu.cu",
                    "src/sampling.cpp",
                    "src/sampling_gpu.cu",
                ],
            ),
        ]

    setup(
        name="pcdet",
        version=version,
        description="OpenPCDet is a general codebase for 3D object detection from point cloud",
        install_requires=[
            "numpy",
            "torch>=1.1",
            "spconv",
            "numba",
            "tensorboardX",
            "easydict",
            "pyyaml",
        ],
        author="Shaoshuai Shi",
        author_email="shaoshuaics@gmail.com",
        license="Apache License 2.0",
        packages=find_packages(exclude=["tools", "data", "output"]),
        cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
        ext_modules=ext_modules,
    )