import os
import copy
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataflow.core.operator import OperatorABC
from dataflow.pipeline.nodes import OperatorNode, KeyNode
from dataflow.wrapper.auto_op import AutoOP, OPRuntime
from dataflow.utils.storage import DataFlowStorage
from dataflow.core import OPERATOR_CLASSES, LLM_SERVING_CLASSES
import atexit
from datetime import datetime
from dataflow.logger import get_logger
import colorsys
class PipelineABC(ABC):
    def __init__(self):
        # list of dict, contains `OPRuntime` class and parameters for `operator.run()`
        self.op_runtimes:list[OPRuntime] = [] 
        self.compiled = False
        # accumulated keys in each operators, index 0 refers to the keys before the first operator
        self.accumulated_keys = []  # list of lists, each sublist contains keys before the operator
        
        # other items
        self.logger = get_logger()
        self.active_llm_serving = None
        # self.serving_resources = defaultdict(dict)
        # self.serving_reference_counter = Counter()
        
        self.op_nodes_list : list[OperatorNode] = []
        self.llm_serving_list = [] # list of LLMServing objects
        self.llm_serving_counter = Counter() # count of LLMServing objects
    
    @abstractmethod
    def forward(self):
        """
        Main Function to run the pipeline
        """
        pass
    
    def compile(self):
        self.compiled = True # flag the pipeline as compiled
        for k, v in vars(self).items():
            if isinstance(v, OperatorABC):
                setattr(self, k, AutoOP(v, k, self)) 
        self.forward()
        # after call forward, call back function in AutoOP will add the OPRuntime object to self.op_runtimes
        
        self.forward = self._compiled_forward
        self.logger.info(
            f"Compiling pipeline and validating key integrity "
            f"across {len(self.op_runtimes)} operator runtimes."
        )
        self._build_operator_nodes_graph()
        # self._draw_graph_for_operators()
        # self._build_serving_resources_map()
        
    def _build_operator_nodes_graph(self):
        """
        Build a graph of operator nodes, each node contains the operator object and its storage.
        """
        for op_runtime in self.op_runtimes:
            llm_serving_obj, storage_obj = None, None
            # get llm_serving object from the operator
            for _, v in vars(op_runtime.op).items():
                if isinstance(v, LLM_SERVING_CLASSES):
                    llm_serving_obj = v
            # get storage object from the function dict
            storage_obj = op_runtime.kwargs.pop("storage", None)
            
            assert isinstance(storage_obj, DataFlowStorage), f"Storage must be a DataFlowStorage object, but got {type(storage_obj)} in {op_runtime}'s `run` function with key `storage`."
            
            # create an operator node
            op_node = OperatorNode(
                op_obj=op_runtime.op,
                op_name=op_runtime.op_name,
                storage=storage_obj,
                llm_serving=llm_serving_obj,
                **op_runtime.kwargs
            )
            
            # append to lists, if None, just keep it
            self.op_nodes_list.append(op_node)
            self.llm_serving_list.append(llm_serving_obj)
            if llm_serving_obj is not None:
                self.llm_serving_counter[llm_serving_obj] += 1
        self.logger.debug(f"Built operator nodes graph with {self.op_nodes_list} nodes, \nand {self.llm_serving_list} LLM Serving objects.")
        
        # get keys in the first storage:
        first_op = self.op_nodes_list[0] if self.op_nodes_list else None
        if first_op and first_op.storage:
            iter_storage_keys = first_op.storage.get_keys_from_dataframe()
        else:
            iter_storage_keys = []

        # print("start keys", iter_storage_keys)

        # all keys in the first storage will be the initial keys for validation
        self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))

        error_msg = []
        # build graph of all operators and keys from all states
        for op_node in self.op_nodes_list:
            # check if accumulated_keys have the input keys of this operator
            # print(op_node, op_node.input_keys, op_node.output_keys)
            for input_key in op_node.input_keys:
                if input_key not in self.accumulated_keys[-1]:
                    error_msg.append(
                        {
                            "input_key": input_key,
                            "op_name": op_node.op_name,
                            "class_name": op_node.op_obj.__class__.__name__,
                            "key_para_name": op_node.input_key_nodes[input_key].key_para_name
                        }
                    )

            # add output keys to accumulated keys
            for output_key in op_node.output_keys:
                if output_key not in iter_storage_keys:
                    iter_storage_keys.append(output_key)
            self.accumulated_keys.append(copy.deepcopy(iter_storage_keys))
        if len(error_msg) != 0:
            # final_error_str = "KeyError in following Operators during pipeline.compile():"
            details = "\n".join(
                f"- Input key '{e['input_key']}' in `{e['op_name']}` "
                f"(class <{e['class_name']}>) does not match any output keys "
                f"from previous operators or dataset keys. "
                f"Check parameter '{e['key_para_name']}' in the `{e['op_name']}.run()`."
                for e in error_msg
            )
            msg = f"Key Matching Error in following Operators during pipeline.compile():\n{details}"
            self.logger.warning(msg)
            raise KeyError(msg)  

        self.final_keys = copy.deepcopy(iter_storage_keys)

        
        for i, keys in enumerate(self.accumulated_keys):
            # print(i, keys)
            pass
        self.logger.debug(f"Accumulated keys after building graph: {self.accumulated_keys}")

        self.input_dataset_node = OperatorNode(
            None,
            "DATASET-INPUT",
            None,
            None,

        )
        self.input_dataset_node.init_output_keys_nodes(self.accumulated_keys[0])
        self.op_nodes_list.insert(0, self.input_dataset_node)
        
        self.output_dataset_node = OperatorNode(
            None,
            "DATASET-OUTPUT",
            None,
            None,
        )
        self.output_dataset_node.init_input_keys_nodes(self.final_keys)
        self.op_nodes_list.append(self.output_dataset_node)

        # set a default dict for all keys
        self.last_modified_index_of_keys: dict[list] = {}
        for key in self.final_keys:
            self.last_modified_index_of_keys[key] = []
        # print(self.last_modified_index_of_keys)

        # now the first op node is THEDATASET op
        for idx, i_op in enumerate(self.op_nodes_list):
            # check for input keys
            for input_key in i_op.input_keys:
                current_keynode:KeyNode = i_op.input_key_nodes[input_key]
                current_keynode.set_index(idx)

                if len(self.last_modified_index_of_keys[input_key]) > 0:
                    last_modified_idx = self.last_modified_index_of_keys[input_key][-1]
                    last_modified_keynode:KeyNode = self.op_nodes_list[last_modified_idx].output_keys_nodes[input_key]
                    # double side ptr for each nodes
                    last_modified_keynode.ptr.append(current_keynode)
                    current_keynode.ptr.append(last_modified_keynode)
            # check for output keys
            for output_key in i_op.output_keys:
                current_keynode:KeyNode = i_op.output_keys_nodes[output_key]
                current_keynode.set_index(idx)
                self.last_modified_index_of_keys[output_key].append(idx)

        for key, value in self.last_modified_index_of_keys.items():
            # print(key, value)
            pass
                    
        for op in self.op_nodes_list:
            # print(op)
            self.logger.debug(f"Operator Node: {op}")
            pass

    # deprecated, use `draw_graph` instead, archived for compatibility
    def _draw_graph_for_operators(self): 
        raise DeprecationWarning(
            "The `_draw_graph_for_operators` method is deprecated. "
            "Please use `draw_graph` method instead for better visualization.")
    
        def _get_op_node_str(self, node:OperatorNode):
            input_keys_string = ""
            for i_key_node in node.input_key_nodes.values():
                input_keys_string += f"\n{i_key_node.key_para_name}={i_key_node.key}"
            output_keys_string = ""
            for o_key_node in node.output_keys_nodes.values():
                output_keys_string += f"\n{o_key_node.key_para_name}={o_key_node.key}"
            # return f"{node.op_name}\n{node.op_obj.__class__.__name__}\n{node.llm_serving.__class__.__name__ if node.llm_serving else 'None'}\n{input_keys_string}\n --- \n{output_keys_string}"
            return f"{node.op_name}\n{node.op_obj.__class__.__name__}\n"

        try:
            import networkx
        except:
            raise ImportError("Please install networkx to draw graph. Please run `pip install networkx[default]`.")
        import matplotlib.pyplot as plt
        G = networkx.DiGraph()
        # add OP nodes
        for op_node in self.op_nodes_list:
            G.add_node(op_node, label=_get_op_node_str(self, op_node))
        # add edges between OP nodes
        for op_node in self.op_nodes_list:
            for output_key_nodes in op_node.output_keys_nodes.values():
                for ptr_key_node in output_key_nodes.ptr:
                    G.add_edge(op_node, self.op_nodes_list[ptr_key_node.index], label=ptr_key_node.key)

        # draw the figure
        pos = networkx.spring_layout(G)
        # pos = networkx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
        # pos = networkx.kamada_kawai_layout(G)


        # pos = networkx.spectral_layout(G)
        # 设置画布大小
        num_nodes = len(G.nodes)
        plt.figure(figsize=(max(10, num_nodes * 0.5), max(8, num_nodes * 0.5)))

        # 绘制图形，使用自定义标签
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        networkx.draw(G, pos, labels=labels, with_labels=True, node_size=1000, node_shape='s', node_color='lightblue', edge_color='gray', arrows=True)

        # 绘制边的标签
        edge_labels = networkx.get_edge_attributes(G, 'label')
        networkx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        # 保存图形
        plt.savefig("operators_graph.png", bbox_inches='tight')
        plt.show()

    def draw_graph(
            self, 
            port=0,
            hide_no_changed_keys=True
            ):
        # compile check
        if not self.compiled:
            self.logger.error("Pipeline is not compiled yet. Please call `compile()` before drawing the graph.")
            raise RuntimeError("Pipeline is not compiled yet. Please call `compile()` before drawing the graph.")
        # import check if pyvis is installed
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("Please install pyvis to draw graph of current pipeline. Please run `pip install pyvis`.")
        

        def _get_op_node_str(node, step = None):
            op_class_name = node.op_obj.__class__.__name__ if node.op_obj.__class__.__name__ !="NoneType" else "Storage/No-Op"
            if step is not None:
                return f"{node.op_name}\n<{op_class_name}>\n(step={step})\n"
            else:
                return f"{node.op_name}\n<{op_class_name}>\n"

        def _get_op_node_title(node):
            input_keys_string = ""
            op_class_name = node.op_obj.__class__.__name__ if node.op_obj.__class__.__name__ !="NoneType" else "Storage/No-Op"
            if op_class_name == "Storage/No-Op":
                for i_key_node in node.input_key_nodes.values():
                    input_keys_string += f"  {i_key_node.key}\n"
                output_keys_string = ""
                for o_key_node in node.output_keys_nodes.values():
                    output_keys_string += f"  {o_key_node.key}\n"
            else:
                for i_key_node in node.input_key_nodes.values():
                    input_keys_string += f"  {i_key_node.key_para_name}={i_key_node.key}\n"
                output_keys_string = ""
                for o_key_node in node.output_keys_nodes.values():
                    output_keys_string += f"  {o_key_node.key_para_name}={o_key_node.key}\n"

            if input_keys_string == "":
                input_keys_string = "  None\n"
            if output_keys_string == "":
                output_keys_string = "  None\n"
            return (
                f"Attrbute: {node.op_name}\n"
                f"Class: {op_class_name}\n"
                f"------\n"
                f"Input:\n {input_keys_string}"
                f"------\n"
                f"Output:\n {output_keys_string}"
            )

        def _hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        def _rgb_to_hex(rgb):
            return "#{:02x}{:02x}{:02x}".format(*rgb)

        def _lerp_color(c1, c2, t):
            return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

        def _step_to_color(step, total_steps):
            """
            冷淡风渐变: 灰蓝 → 灰紫 → 冷蓝
            """
            # start = _hex_to_rgb("#bfdefc")  # 浅钢蓝
            # mid   = _hex_to_rgb("#aaa1d3")  # 灰紫
            # end   = _hex_to_rgb("#888794")  # 冷蓝灰
            # start = _hex_to_rgb("#FCF1D0")  # 浅钢蓝
            # mid   = _hex_to_rgb("#DBFFDD")  # 灰紫
            # end   = _hex_to_rgb("#C8E0F9")  # 冷蓝灰
            start = _hex_to_rgb("#D5B4EC")  # 浅钢蓝
            mid   = _hex_to_rgb("#879DF8")  # 灰紫
            end   = _hex_to_rgb("#81CDF9")  # 冷蓝灰

            if total_steps <= 1:
                return _rgb_to_hex(start)
            
            mid_point = (total_steps - 1) / 2
            if step <= mid_point:
                t = step / mid_point
                rgb = _lerp_color(start, mid, t)
            else:
                t = (step - mid_point) / mid_point
                rgb = _lerp_color(mid, end, t)
            
            return _rgb_to_hex(rgb)
        
        # def _step_to_color(step, total_steps):
        #     # 红色 (255, 0, 0) → 蓝色 (0, 0, 255)
        #     r_start, g_start, b_start = (255, 0, 0)
        #     r_end, g_end, b_end = (0, 0, 255)
        #     t = step / max(total_steps - 1, 1)  # 归一化到 [0, 1]
        #     r = int(r_start + (r_end - r_start) * t)
        #     g = int(g_start + (g_end - g_start) * t)
        #     b = int(b_start + (b_end - b_start) * t)
        #     return f"#{r:02x}{g:02x}{b:02x}"

        # def _step_to_color(step, total_steps):
        #     """
        #     从红 → 紫 → 蓝的平滑渐变
        #     """
        #     if total_steps <= 1:
        #         return "#ff0000"  # 只有一个节点时直接红色
            
        #     mid_point = (total_steps - 1) / 2
        #     if step <= mid_point:
        #         # 红(0°) → 紫(300°)
        #         h_start, h_end = 0 / 360, 300 / 360
        #         t = step / mid_point
        #     else:
        #         # 紫(300°) → 蓝(240°)
        #         h_start, h_end = 300 / 360, 240 / 360
        #         t = (step - mid_point) / mid_point
            
        #     # 饱和度和亮度固定高值
        #     s, l = 1.0, 0.5
        #     h = h_start + (h_end - h_start) * t
        #     r, g, b = colorsys.hls_to_rgb(h, l, s)
            
        #     return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

        # 生成 PyVis 图
        net = Network(height="800px", width="100%", directed=True)
        net.force_atlas_2based()
        net.toggle_physics(True)
        net.set_options("""
        {
            "physics": {
                "forceAtlas2Based": {
                    "springLength": 300,
                    "springConstant": 0.01
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            }
        }
        """)


        for idx, op_node in enumerate(self.op_nodes_list):
            node_color = _step_to_color(idx, len(self.op_nodes_list))
            net.add_node(
                n_id=id(op_node),
                label=_get_op_node_str(op_node, step=idx),
                title=_get_op_node_title(op_node),
                color=node_color,
                shape="box"
            )

        for op_node in self.op_nodes_list:
            for output_key_nodes in op_node.output_keys_nodes.values():
                for ptr_key_node in output_key_nodes.ptr:
                    target_node = self.op_nodes_list[ptr_key_node.index]
                    if hide_no_changed_keys and op_node==self.op_nodes_list[0] and target_node==self.op_nodes_list[-1]:
                        # hide the keys that are not changed from input dataset to first operator
                        continue

                    net.add_edge(
                        source=id(op_node),
                        to=id(target_node),
                        label=ptr_key_node.key,
                        color="gray"
                    )


        # Timestamped filename to avoid overwriting
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        os.makedirs(".pyvis", exist_ok=True)
        output_html = os.path.abspath(os.path.join(".pyvis" ,f"operators_graph_{ts}.html"))
        net.save_graph(output_html)

        # Automatically delete the file on exit (whether normal exit or Ctrl-C)
        def _cleanup():
            try:
                if os.path.exists(output_html):
                    os.remove(output_html)
                    print(f"🧹 Deleted temp file: {output_html}")
            except Exception as e:
                print(f"Failed to clean up file: {e}")
        atexit.register(_cleanup)

        # Select port
        if port == 0:
            sock = socket.socket()
            sock.bind(('', 0))
            port = sock.getsockname()[1]
            sock.close()

        # Start HTTP service (main thread blocking)
        class SilentHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        # Change to the directory where the file is located so the static service can find it
        orig_cwd = os.getcwd()
        try:
            serve_dir = os.path.dirname(output_html) or "."
            os.chdir(serve_dir)
            url = f"http://localhost:{port}/{os.path.basename(output_html)}"
            print(f"✅ Graph generated, access it at: {url}")


            with HTTPServer(('0.0.0.0', port), SilentHandler) as httpd:
                print(f"HTTP service started, listening on port {port} (Ctrl-C to exit, HTML file will be deleted on exit)")
                try:
                    webbrowser.open(url)
                    httpd.serve_forever()
     
                except KeyboardInterrupt:
                    print("\n❌ Interrupt signal received, exiting and cleaning up file...")
        except Exception as e:
            print(f"❌ Failed to start HTTP service: {e}")
            self.logger.error(f"Failed to start HTTP service: {e}")
        finally:
            os.chdir(orig_cwd)

            # atexit 会负责删除文件，这里无需重复删除
        # # 保存 HTML
        # output_html = "operators_graph.html"
        # net.save_graph(output_html)

        # # 选择端口
        # if port == 0:
        #     sock = socket.socket()
        #     sock.bind(('', 0))
        #     port = sock.getsockname()[1]
        #     sock.close()

        # # 启动 HTTP 服务（主线程）
        # class SilentHandler(SimpleHTTPRequestHandler):
        #     def log_message(self, format, *args):
        #         pass

        # os.chdir(os.path.dirname(os.path.abspath(output_html)))
        # url = f"http://localhost:{port}/{output_html}"
        # print(f"✅ 图已生成，访问: {url}")
        # webbrowser.open(url)

        # # 阻塞运行直到 Ctrl-C
        # try:
        #     with HTTPServer(('0.0.0.0', port), SilentHandler) as httpd:
        #         print(f"HTTP 服务已启动，监听端口 {port}，按 Ctrl-C 退出")
        #         httpd.serve_forever()
        # except KeyboardInterrupt:
        #     print("\n❌ 已退出可视化服务")

    # def _build_serving_resources_map(self):
    #     for op_runtime in self.op_runtimes:
    #         for _, v in vars(op_runtime.op).items():
    #             if isinstance(v, LLMServingABC):
    #                 self.serving_resources[op_runtime.op]["LLMServingABC"] = v
    #                 self.serving_reference_count[v] += 1
                    
    def _compiled_forward(self):
        # for loop for each op and its `storage` status
        for op_node in self.op_nodes_list:
            self.logger.debug(f"Ready to run {op_node}, with serving={op_node.llm_serving}, active_llm_serving={self.active_llm_serving}")
            if op_node.llm_serving != None:
                if self.active_llm_serving and self.active_llm_serving is not op_node.llm_serving:
                    self.logger.debug(f"Detected active LLM Serving {self.active_llm_serving}, new serving {op_node.llm_serving}, cleaning up...")
                    self.active_llm_serving.cleanup()
                self.active_llm_serving = op_node.llm_serving

                op_node.op_obj.run(
                    storage=op_node.storage,
                    **op_node.kwargs
                )
            if op_node.llm_serving != None:
                self.llm_serving_counter[self.active_llm_serving] -= 1
                if self.llm_serving_counter[self.active_llm_serving] == 0:
                    self.logger.debug(f"Detected LLM Serving {self.active_llm_serving} ref reduced to 0, cleaning up...")
                    self.active_llm_serving.cleanup()
                    self.active_llm_serving = None
