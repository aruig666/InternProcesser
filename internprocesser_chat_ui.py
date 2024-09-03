import copy
import pathlib
import warnings
from dataclasses import asdict, dataclass
from typing import Callable, List, Optional

from utils.visualization import create_3D_Viewer
import streamlit as st
import torch
from torch import nn
from transformers.generation.utils import (LogitsProcessorList,
                                           StoppingCriteriaList)
from transformers.utils import logging

from transformers import AutoTokenizer, AutoModelForCausalLM  # isort: skip

logger = logging.get_logger(__name__)



# 模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('agem0402/internlmprocesser')

# model_name_or_path = "/root/InternLM/XTuner/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
model_name_or_path = model_dir

@dataclass
class GenerationConfig:
    # this config is used for chat to provide more diversity
    max_length: int = 2048
    top_p: float = 0.75
    temperature: float = 0.1
    do_sample: bool = True
    repetition_penalty: float = 1.000


@torch.inference_mode()
def generate_interactive(
    model,
    tokenizer,
    prompt,
    generation_config: Optional[GenerationConfig] = None,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor],
                                                List[int]]] = None,
    additional_eos_token_id: Optional[int] = None,
    **kwargs,
):
    inputs = tokenizer([prompt], padding=True, return_tensors='pt')
    input_length = len(inputs['input_ids'][0])
    for k, v in inputs.items():
        inputs[k] = v.cuda()
    input_ids = inputs['input_ids']
    _, input_ids_seq_length = input_ids.shape[0], input_ids.shape[-1]
    if generation_config is None:
        generation_config = model.generation_config
    generation_config = copy.deepcopy(generation_config)
    model_kwargs = generation_config.update(**kwargs)
    bos_token_id, eos_token_id = (  # noqa: F841  # pylint: disable=W0612
        generation_config.bos_token_id,
        generation_config.eos_token_id,
    )
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    if additional_eos_token_id is not None:
        eos_token_id.append(additional_eos_token_id)
    has_default_max_length = kwargs.get(
        'max_length') is None and generation_config.max_length is not None
    if has_default_max_length and generation_config.max_new_tokens is None:
        warnings.warn(
            f"Using 'max_length''s default ({repr(generation_config.max_length)}) \
                to control the generation length. "
            'This behaviour is deprecated and will be removed from the \
                config in v5 of Transformers -- we'
            ' recommend using `max_new_tokens` to control the maximum \
                length of the generation.',
            UserWarning,
        )
    elif generation_config.max_new_tokens is not None:
        generation_config.max_length = generation_config.max_new_tokens + \
            input_ids_seq_length
        if not has_default_max_length:
            logger.warn(  # pylint: disable=W4902
                f"Both 'max_new_tokens' (={generation_config.max_new_tokens}) "
                f"and 'max_length'(={generation_config.max_length}) seem to "
                "have been set. 'max_new_tokens' will take precedence. "
                'Please refer to the documentation for more information. '
                '(https://huggingface.co/docs/transformers/main/'
                'en/main_classes/text_generation)',
                UserWarning,
            )

    if input_ids_seq_length >= generation_config.max_length:
        input_ids_string = 'input_ids'
        logger.warning(
            f"Input length of {input_ids_string} is {input_ids_seq_length}, "
            f"but 'max_length' is set to {generation_config.max_length}. "
            'This can lead to unexpected behavior. You should consider'
            " increasing 'max_new_tokens'.")

    # 2. Set generation parameters if not already defined
    logits_processor = logits_processor if logits_processor is not None \
        else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None \
        else StoppingCriteriaList()

    logits_processor = model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=input_ids_seq_length,
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        logits_processor=logits_processor,
    )

    stopping_criteria = model._get_stopping_criteria(
        generation_config=generation_config,
        stopping_criteria=stopping_criteria)
    logits_warper = model._get_logits_warper(generation_config)

    unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
    scores = None
    while True:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)
        # forward pass to get next token
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        next_token_scores = logits_warper(input_ids, next_token_scores)

        # sample
        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if generation_config.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=False)
        unfinished_sequences = unfinished_sequences.mul(
            (min(next_tokens != i for i in eos_token_id)).long())

        output_token_ids = input_ids[0].cpu().tolist()
        output_token_ids = output_token_ids[input_length:]
        for each_eos_token_id in eos_token_id:
            if output_token_ids[-1] == each_eos_token_id:
                output_token_ids = output_token_ids[:-1]
        response = tokenizer.decode(output_token_ids)

        yield response
        # stop when each sentence is finished
        # or if we exceed the maximum length
        if unfinished_sequences.max() == 0 or stopping_criteria(
                input_ids, scores):
            break


def on_btn_click():
    del st.session_state.messages


@st.cache_resource
def load_model():
    model = (AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                              trust_remote_code=True)
    return model, tokenizer


def prepare_generation_config():
    with st.sidebar:
        max_length = st.slider('Max Length',
                               min_value=8,
                               max_value=32768,
                               value=2048)
        top_p = st.slider('Top P', 0.0, 1.0, 0.75, step=0.01)
        temperature = st.slider('Temperature', 0.0, 1.0, 0.1, step=0.01)
        st.button('Clear Chat History', on_click=on_btn_click)

    generation_config = GenerationConfig(max_length=max_length,
                                         top_p=top_p,
                                         temperature=temperature)

    return generation_config


user_prompt = '<|im_start|>user\n{user}<|im_end|>\n'
robot_prompt = '<|im_start|>assistant\n{robot}<|im_end|>\n'
cur_query_prompt = '<|im_start|>user\n{user}<|im_end|>\n\
    <|im_start|>assistant\n'


def combine_history(prompt):
    messages = st.session_state.messages
    meta_instruction = ('')
    total_prompt = f"<s><|im_start|>system\n{meta_instruction}<|im_end|>\n"
    for message in messages:
        cur_content = message['content']
        if message['role'] == 'user':
            cur_prompt = user_prompt.format(user=cur_content)
        elif message['role'] == 'robot':
            cur_prompt = robot_prompt.format(robot=cur_content)
        else:
            raise RuntimeError
        total_prompt += cur_prompt
    total_prompt = total_prompt + cur_query_prompt.format(user=prompt)
    return total_prompt


def main():
    # torch.cuda.empty_cache()
    print('load model begin.')
    model, tokenizer = load_model()
    print('load model end.')


    st.title('InternPrecesser')
    generation_config = prepare_generation_config()

    step_dir = pathlib.Path('./uploads')
    if st.button("选择铣削测试零件"):
        stl_file = step_dir.joinpath("millingpart.stl")
        html_file = step_dir.joinpath("millingpart.html")
        create_3D_Viewer(stl_file, html_file)
        st.markdown(
      "<h2 style='font-size:24px; color:blue;'>复制问题，输入对话框</h2>", 
      unsafe_allow_html=True
        )
        # st.title("复制问题，输入对话框")
        st.write("询问工艺")
        st.empty()
        q1 = '''
    零件限位块，是一个铣削零件，
    长40.0毫米、宽15.0毫米、高15.0毫米，体积为7850.97立方毫米，表面积为3459.11平方毫米。
    在[0.0, 0.0, 1.0]方向下有4个特征。 
        第1个特征为上表面特征，面积600.00平方毫米，体积1200.00立方毫米，深度2.00毫米。 
        第2个特征为封闭槽特征，面积144.95平方毫米，体积220.60立方毫米，深度4.40毫米。 
        第3个特征为封闭槽特征，面积144.95平方毫米，体积220.60立方毫米，深度4.40毫米。 
        第4个特征为开放腔体特征，面积1759.41平方毫米，体积0.00立方毫米，深度15.00毫米。 
    在[0.0, 1.0, 0.0]方向下有5个特征。 
    第1个特征为封闭槽特征，面积98.17平方毫米，体积101.44立方毫米，深度5.20毫米。 
    第2个特征为封闭槽特征，面积98.17平方毫米，体积101.44立方毫米，深度5.20毫米。 
    第3个特征为开放腔体特征，面积564.76平方毫米，体积0.00立方毫米，深度15.00毫米。 
    第4个特征为封闭腔体特征，面积61.58平方毫米，体积29.57立方毫米，深度9.80毫米。 
    第5个特征为封闭腔体特征，面积61.58平方毫米，体积29.57立方毫米，深度9.80毫米。 
    这个零件应该如何加工？
        '''
        st.write(q1)
        st.write("询问报价")
        st.empty()
        q2 = '''
    零件的坯料体积为12138.00立方毫米,面积3459.11平方毫米，材料为铝合金板材，
    已知铝合金板材的市场价为20元每公斤，加工中心的工时单价为90元每小时。
    总加工工时246.11秒，零件报价多少合适？
        '''
        st.write(q2)
    if st.button("选择车削测试零件"):
        stl_file = step_dir.joinpath("turningpart.stl")
        html_file = step_dir.joinpath("turningpart.html")
        create_3D_Viewer(stl_file, html_file)
        st.write("复制问题，输入对话框")
        st.write("询问工艺")
        st.empty()
        q1 = '''
    零件导向轴，是一个车削零件,  \n
    长25.0毫米、宽25.0毫米、高103.5毫米，体积为33140.08立方毫米，表面积为7516.08平方毫米，
    这个零件有27个回转面，2铣削面。 
    回转面的面积为162.39平方毫米，体积为252.54立方毫米。 
    第1个面的面积为162.39平方毫米，体积为252.54立方毫米。 
    第2个面的面积为137.84平方毫米，体积为195.00立方毫米。 
    这个车削零件应该如何加工？
        '''
        st.write(q1)
        st.write("询问报价")
        st.empty()
        q2 = '''
    车削零件的坯料体积为60404.58立方毫米,面积7516.08平方毫米，材料为铝合金，
    已知铝合金棒材的市场价为30元每公斤,数控车床的工时单价为60元每小时，
    总加工工时11424.30秒，零件报价多少合适？
        '''
        st.write(q2)
    if st.button("选择钣金测试零件"):
        succeed = True
        visual = False
        stl_file = step_dir.joinpath("turningpart.stl")
        html_file = step_dir.joinpath("turningpart.html")
        create_3D_Viewer(stl_file, html_file)
        st.title("复制问题，输入对话框")
        st.write("询问工艺")
        st.empty()
        q1 = '''
    零件连接支架，是一个钣金零件,
    长600.0毫米、宽75.0毫米，高35.0毫米，所使用板材厚度3.0毫米.
    这个零件外轮廓长度为1401.55毫米，
    普通折弯1，长度分别为600.0毫米，
    刨槽5个，
    这个钣金零件应该如何加工？
        '''
        st.write(q1)
        st.write("询问报价")
        st.empty()
        q2 = '''
    钣金零件的坯料厚度为3.0毫米，体积为187174.84立方毫米,材料为铝合金。
    3.0毫米铝合金薄板的市场价为120元每平方米，
    零件的切割轮廓为1401.55毫米，1个普通折弯，5个刨槽，
    板材切割单价1.2元每米，穿刺每个0.1元，普通折弯每个2元，折死边每个5元，刨槽每个4元，
    零件报价多少合适？
        '''
        st.write(q2)

    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message['role'], avatar=message.get('avatar')):
            st.markdown(message['content'])

    # Accept user input
    if prompt := st.chat_input('What is up?'):
    # if prompt := st.chat_input(q1):
        # Display user message in chat message container
        with st.chat_message('user'):
            st.markdown(prompt)
        real_prompt = combine_history(prompt)
        # Add user message to chat history
        st.session_state.messages.append({
            'role': 'user',
            'content': prompt,
        })

        with st.chat_message('robot'):
            message_placeholder = st.empty()
            for cur_response in generate_interactive(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=real_prompt,
                    additional_eos_token_id=92542,
                    **asdict(generation_config),
            ):
                # Display robot response in chat message container
                message_placeholder.markdown(cur_response + '▌')
            message_placeholder.markdown(cur_response)
        # Add robot response to chat history
        st.session_state.messages.append({
            'role': 'robot',
            'content': cur_response,  # pylint: disable=undefined-loop-variable
        })
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
