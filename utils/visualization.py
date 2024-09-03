import pathlib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import pyvista as pv
from pyvista import examples
import os


def create_3D_Viewer(stl_file: pathlib.Path, html_file: pathlib.Path):
    ## Initialize pyvista reader and plotter
    # reader = pv.STLReader(str(stl_file))
    # plotter = pv.Plotter(
    #     border=True,
    #     window_size=[580, 400])
    # plotter.background_color = "white"

    # ## Read data and send to plotter
    # mesh = reader.read()
    # plotter.add_mesh(mesh, color='white')

    ## Export to an external pythreejs
    model_html = str(html_file)
    # other = plotter.export_html(model_html)

    ## Read the exported model
    with open(model_html, 'r', encoding='utf-8', errors='ignore') as file:
        model = file.read()

    ## Show in webpage
    components.html(model, height=300)


def delect_history_files():
    delete_path = pathlib.Path('./uploads')
    # 创建一个删除文件的按钮
    if st.button("删除文件"):
        delectfiles = list(delete_path.rglob("*"))
        for delete_file in delectfiles:
            try:
                # 删除文件
                os.remove(str(delete_file))
                st.success(f"文件 {delete_path.name} 已被删除。")
            except Exception as e:
                st.error(f"删除文件时出错: {e}")
        # st.experimental_rerun()  # 刷新页面
