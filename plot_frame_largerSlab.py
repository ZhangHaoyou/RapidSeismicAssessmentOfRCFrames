import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_opaque_cube(ax, x=10, y=20, z=30, dx=40, dy=50, dz=60, color="grey"):

    xx = np.linspace(x, x+dx, 2)
    yy = np.linspace(y, y+dy, 2)
    zz = np.linspace(z, z+dz, 2)

    xx2, yy2 = np.meshgrid(xx, yy)
    ax.plot_surface(xx2, yy2, np.full_like(xx2, z), color=color)
    ax.plot_surface(xx2, yy2, np.full_like(xx2, z+dz), color=color)

    yy2, zz2 = np.meshgrid(yy, zz)
    ax.plot_surface(np.full_like(yy2, x), yy2, zz2, color=color)
    ax.plot_surface(np.full_like(yy2, x+dx), yy2, zz2, color=color)

    xx2, zz2 = np.meshgrid(xx, zz)
    ax.plot_surface(xx2, np.full_like(yy2, y), zz2, color=color)
    ax.plot_surface(xx2, np.full_like(yy2, y+dy), zz2, color=color)

def plot_columns(ax, z0, hs, nls, lls, nss, lss, bc, ts, color_col):

    for ix in range(nls+1):
        lx = ix*lls
        for iy in range(nss+1):
            ly = iy*lss
            plot_opaque_cube(ax=ax, x=lx, y=ly, z=z0, dx=bc, dy=bc, dz=hs, color=color_col)

def plot_beams(ax, z0, hs, nls, lls, nss, lss, bc, hb, bb, ts, color_beam):

    # plot the vertical beams
    for ix in range(nls+1):
        lx = ix*lls
        for iy in range(nss):
            ly = iy*lss
            plot_opaque_cube(ax=ax, x=lx, y=ly+bc, z=z0+hs-hb, dx=bb, dy=lss-bc, dz=hb-ts, color=color_beam)
    # plot the horizontal beams
    for iy in range(nss+1):
        ly = iy*lss
        for ix in range(nls):
            lx = ix*lls
            plot_opaque_cube(ax=ax, x=lx+bc, y=ly, z=z0+hs-hb, dx=lls-bc, dy=bb, dz=hb-ts, color=color_beam)

def plot_story(ax, z0, hs, nls, lls, nss, lss, bc, hb, bb, ts, color_col, color_beam, color_slab):

    plot_opaque_cube(ax=ax, x=0, y=0, z=z0+hs-ts, dx=nls*lls+bc, dy=nss*lss+bc, dz=ts, color=color_slab)
    plot_beams(ax=ax, z0=z0, hs=hs, nls=nls, lls=lls, nss=nss, lss=lss, bc=bc, hb=hb, bb=bb, ts=ts, color_beam=color_beam)
    plot_columns(ax=ax, z0=z0, hs=hs-ts, nls=nls, lls=lls, nss=nss, lss=lss, bc=bc, ts=ts, color_col=color_col)

def crop_img(img_path="./frame.jpg", x0=0.356, y0=0.25):
    x1 = 1-x0
    y1 = 1-y0
    img    = cv2.imread(img_path)
    shape  = img.shape
    height = shape[0]
    width  = shape[1]
    croped_img = img[int(y0*height):int(y1*height), int(x0*width):int(x1*width)]
    cv2.imwrite("./frame_croped.jpg", croped_img)

def plot_frame(ns=3, hs=3.33, nss=2, lss=5.4, nls=4, lls=4.1, bc=0.5, hb=0.4, bb=0.3, ts=0.18,
    figsize=(2.5,0.5), path="./frame.jpg", color_col="grey", color_beam="green", color_slab="orange"):

    fig = plt.figure(figsize=figsize)
    ax  = fig.add_subplot(1, 1, 1, projection="3d")

    for story in range(ns):
        z0 = story*hs
        plot_story(ax=ax, z0=z0, hs=hs, nls=nls, lls=lls, nss=nss, lss=lss, bc=bc, hb=hb, bb=bb, ts=ts,
            color_col=color_col, color_beam=color_beam, color_slab=color_slab)
    plt.gca().set_box_aspect((nls*lls+bc, nss*lss+bc, ns*hs))
    ax.view_init(10,35)
    plt.axis('off')
    plt.savefig(path, dpi=800)
    # plt.show()
    plt.close(fig)
    crop_img()
    # print("cropped")


if __name__ == '__main__':
    # plot_opaque_cube()
    # fig = plt.figure(figsize=(3,2))
    # ax  = fig.add_subplot(1, 1, 1, projection="3d")
    plot_frame()
    # crop_img()
