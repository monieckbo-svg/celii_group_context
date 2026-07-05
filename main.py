import datetime
import traceback
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")  # 消息时间戳锚定北京
import uuid
from collections import defaultdict
from typing import Optional, List, Tuple

from astrbot.api.event import filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register
from astrbot.api.provider import ProviderRequest, LLMResponse, Provider
from astrbot.api import logger, AstrBotConfig
from astrbot.api.message_components import At, Image, Plain, Forward, Reply
from astrbot.api.platform import MessageType
import astrbot.api.message_components as Comp
from astrbot.core.utils.io import download_image_by_url

try:
    from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
    IS_AIOCQHTTP = True
except ImportError:
    IS_AIOCQHTTP = False


CHATROOM_SYSTEM_PROMPT = "You are now in a chatroom. The chat history is as above. Now, new messages are coming. Please react to it."
DEFAULT_CAPTION_PROMPT = "用中文简要描述这张图片的内容，包括文字、人物、场景等关键信息。"


@register("celii_group_context", "celii-astra", "群聊上下文增强：消息收集、图片转述、合并转发、唤醒词、[skip]过滤", "2.0.0")
class GroupContextPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        self.session_chats = defaultdict(list)

        # 合并转发
        self.enable_forward_analysis = bool(self.get_cfg("enable_forward_analysis", True))
        self.forward_prefix = "【合并转发内容】"

        # 图片处理
        self.enable_image_recognition = bool(self.get_cfg("enable_image_recognition", True))
        self.image_caption = bool(self.get_cfg("image_caption", False))
        self.image_caption_provider_id = str(self.get_cfg("image_caption_provider_id", "") or "")
        self.image_caption_prompt = str(self.get_cfg("image_caption_prompt", DEFAULT_CAPTION_PROMPT) or DEFAULT_CAPTION_PROMPT)
        self.image_carry_rounds = int(self.get_cfg("image_carry_rounds", 1))

        # 私聊场景控制
        self.enable_private_control = bool(self.get_cfg("enable_private_control", False))
        self.private_conversation_rounds_limit = int(self.get_cfg("private_conversation_rounds_limit", 10))
        self.private_image_carry_rounds = int(self.get_cfg("private_image_carry_rounds", 5))

        # 指令过滤
        self.enable_command_filter = bool(self.get_cfg("enable_command_filter", True))
        self.command_prefixes = self.get_cfg("command_prefixes", ["/"])

        # 唤醒词
        self.wake_words = self.get_cfg("wake_words", [])
        self.vip_qq = str(self.get_cfg("vip_qq", "") or "")
        self.vip_always_respond = bool(self.get_cfg("vip_always_respond", False))

        # 启动日志
        logger.info("[celii_gc] 插件已初始化")
        logger.info(f"  合并转发: {'开' if self.enable_forward_analysis else '关'}")
        logger.info(f"  图片识别: {'开' if self.enable_image_recognition else '关'}")
        if self.enable_image_recognition:
            mode = '转述描述' if self.image_caption else 'base64注入(警告:贵!)'
            logger.info(f"  图片模式: {mode}")
            if self.image_caption:
                logger.info(f"  转述Provider: {self.image_caption_provider_id or '(使用默认provider)'}")
                logger.info(f"  转述Prompt: {self.image_caption_prompt[:60]}...")
            logger.info(f"  图片携带轮数: {self.image_carry_rounds}")
        if self.wake_words:
            logger.info(f"  唤醒词: {self.wake_words}")
        if self.vip_qq:
            logger.info(f"  VIP: {self.vip_qq}, 始终响应: {self.vip_always_respond}")

    def get_cfg(self, key: str, default=None):
        return self.config.get(key, default)

    def is_command(self, message: str) -> bool:
        if not self.enable_command_filter or not message:
            return False
        message = message.strip()
        for prefix in self.command_prefixes:
            if message.startswith(prefix):
                return True
        return False

    def _check_wake_words(self, event: AstrMessageEvent) -> bool:
        message_text = event.message_str.strip()
        if not message_text:
            return False
        sender_id = str(event.get_sender_id())
        if self.vip_qq and sender_id == self.vip_qq and self.vip_always_respond:
            logger.info(f"[唤醒词] VIP用户 {sender_id}，直接触发")
            return True
        for word in self.wake_words:
            if word in message_text:
                logger.info(f"[唤醒词] 检测到 '{word}'，触发回复")
                return True
        return False

    def _extract_image_url(self, image_data) -> Optional[str]:
        if not image_data:
            return None
        if isinstance(image_data, str):
            return image_data
        if isinstance(image_data, dict):
            if "image_url" in image_data:
                image_url_obj = image_data["image_url"]
                if isinstance(image_url_obj, dict) and "url" in image_url_obj:
                    return image_url_obj["url"]
            if "url" in image_data:
                return image_data["url"]
        if isinstance(image_data, Image):
            if hasattr(image_data, 'url') and image_data.url:
                return image_data.url
            if hasattr(image_data, 'file') and image_data.file:
                return image_data.file
        return None

    # ==================== 合并转发 ====================

    async def _detect_forward_message(self, event) -> Optional[str]:
        if not IS_AIOCQHTTP or not isinstance(event, AiocqhttpMessageEvent):
            return None
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Forward):
                return seg.id
        reply_seg = None
        for seg in event.message_obj.message:
            if isinstance(seg, Comp.Reply):
                reply_seg = seg
                break
        if reply_seg:
            try:
                client = event.bot
                original_msg = await client.api.call_action('get_msg', message_id=reply_seg.id)
                if original_msg and 'message' in original_msg:
                    original_message_chain = original_msg['message']
                    if isinstance(original_message_chain, list):
                        for segment in original_message_chain:
                            if isinstance(segment, dict) and segment.get("type") == "forward":
                                return segment.get("data", {}).get("id")
            except Exception as e:
                logger.error(f"获取回复消息失败: {e}")
        return None

    # ==================== 图片处理 ====================

    async def _encode_image_bs64(self, image_url: str) -> str:
        try:
            import base64
            if image_url.startswith("data:image"):
                return image_url
            elif image_url.startswith("base64://"):
                return image_url.replace("base64://", "data:image/jpeg;base64,")
            elif image_url.startswith("http"):
                image_path = await download_image_by_url(image_url)
                with open(image_path, "rb") as f:
                    image_bs64 = base64.b64encode(f.read()).decode("utf-8")
                return "data:image/jpeg;base64," + image_bs64
            elif image_url.startswith("file:///"):
                image_path = image_url.replace("file:///", "")
                with open(image_path, "rb") as f:
                    image_bs64 = base64.b64encode(f.read()).decode("utf-8")
                return "data:image/jpeg;base64," + image_bs64
            else:
                with open(image_url, "rb") as f:
                    image_bs64 = base64.b64encode(f.read()).decode("utf-8")
                return "data:image/jpeg;base64," + image_bs64
        except Exception as e:
            logger.error(f"图片转base64失败: {image_url[:80]}..., 错误: {e}")
            return ""

    async def get_image_caption(self, image_url: str) -> str:
        """获取图片描述 - 完全模仿AstrBot核心的_request_img_caption"""
        import base64 as b64mod, tempfile, os

        # 1. 获取provider（详细日志）
        provider = None
        logger.info(f"[图片转述] 配置的provider_id='{self.image_caption_provider_id}'")
        
        if self.image_caption_provider_id:
            provider = self.context.get_provider_by_id(self.image_caption_provider_id)
            logger.info(f"[图片转述] get_provider_by_id 结果: {type(provider)}, id={getattr(provider, 'id', '无id')}")
        
        if not provider:
            provider = self.context.get_using_provider()
            logger.info(f"[图片转述] 回退到默认provider: {type(provider)}, id={getattr(provider, 'id', '无id')}")

        if not provider:
            raise Exception("没有可用的Provider")

        # 2. 将图片转为临时文件（provider只认文件路径和http URL）
        tmp_path = None
        need_cleanup = False
        try:
            url_type = image_url[:30] if len(image_url) > 30 else image_url
            logger.info(f"[图片转述] 图片URL类型: {url_type}...")

            if image_url.startswith("data:image"):
                header, b64data = image_url.split(",", 1)
                img_bytes = b64mod.b64decode(b64data)
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.write(fd, img_bytes)
                os.close(fd)
                need_cleanup = True
                logger.info(f"[图片转述] base64解码存文件: {tmp_path}, 大小={len(img_bytes)}字节")
            elif image_url.startswith("base64://"):
                b64data = image_url.replace("base64://", "")
                img_bytes = b64mod.b64decode(b64data)
                fd, tmp_path = tempfile.mkstemp(suffix=".jpg")
                os.write(fd, img_bytes)
                os.close(fd)
                need_cleanup = True
                logger.info(f"[图片转述] base64://解码存文件: {tmp_path}, 大小={len(img_bytes)}字节")
            elif image_url.startswith("http"):
                tmp_path = await download_image_by_url(image_url)
                logger.info(f"[图片转述] HTTP下载到: {tmp_path}")
            else:
                tmp_path = image_url
                logger.info(f"[图片转述] 当作本地路径: {tmp_path}")

            if not tmp_path or not os.path.exists(tmp_path):
                raise Exception(f"图片文件不存在: {tmp_path}")

            file_size = os.path.getsize(tmp_path)
            logger.info(f"[图片转述] 文件确认存在: {tmp_path}, {file_size}字节")

            # 3. 调用provider（完全模仿AstrBot核心，不传session_id和persist）
            prompt = self.image_caption_prompt
            logger.info(f"[图片转述] 开始调用text_chat, prompt长度={len(prompt)}")

            response = await provider.text_chat(
                prompt=prompt,
                image_urls=[tmp_path],
            )

            if not response or not response.completion_text:
                raise Exception("Provider返回空描述")

            logger.info(f"[图片转述] 成功! 描述长度={len(response.completion_text)}")
            return response.completion_text

        finally:
            if need_cleanup and tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    async def _process_image(self, url: str, current_message_content: list, full_text_ref: list) -> None:
        """统一图片处理入口。full_text_ref = [str]，用单元素列表模拟引用传递"""
        full_text = full_text_ref[0]

        if not self.enable_image_recognition:
            full_text += " [图片]"
            full_text_ref[0] = full_text
            return

        if self.image_caption:
            # 转述模式：便宜模型转文字
            try:
                caption = await self.get_image_caption(url)
                full_text += f" [图片描述: {caption}]"
            except Exception as e:
                logger.error(f"[图片转述] 失败: {e}")
                logger.error(f"[图片转述] 堆栈: {traceback.format_exc()}")
                full_text += " [图片]"
            full_text_ref[0] = full_text
        else:
            # base64注入模式
            if full_text:
                current_message_content.append({"type": "text", "text": full_text})
                full_text = ""
            image_data = await self._encode_image_bs64(url)
            if image_data:
                current_message_content.append({"type": "image_url", "image_url": {"url": image_data}})
            else:
                full_text += " [图片]"
            full_text_ref[0] = full_text

    # ==================== 消息处理 ====================

    @filter.platform_adapter_type(filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return

        message_text = ""
        for comp in event.message_obj.message:
            if isinstance(comp, Plain):
                message_text += comp.text

        if self.is_command(message_text):
            return

        has_valid_content = False
        for comp in event.message_obj.message:
            if isinstance(comp, (Plain, Image)):
                has_valid_content = True
                break
            if IS_AIOCQHTTP and isinstance(comp, Forward):
                has_valid_content = True
                break

        if not has_valid_content:
            return

        # 唤醒词检测
        if not event.is_at_or_wake_command:
            if self._check_wake_words(event):
                event.is_at_or_wake_command = True

        try:
            await self.handle_message(event)
        except BaseException as e:
            logger.error(f"记录群聊消息失败: {e}")
            logger.error(traceback.format_exc())

    async def handle_message(self, event: AstrMessageEvent):
        datetime_str = datetime.datetime.now(BEIJING).strftime("%H:%M:%S")
        current_message_content = []
        full_text = f"[{event.message_obj.sender.nickname}/{datetime_str}]: "

        # 1. 合并转发消息
        if self.enable_forward_analysis and IS_AIOCQHTTP:
            forward_id = await self._detect_forward_message(event)
            if forward_id and isinstance(event, AiocqhttpMessageEvent):
                try:
                    client = event.bot
                    forward_data = await client.api.call_action('get_forward_msg', id=forward_id)
                    messages = forward_data.get("messages", [])
                    full_text += f"\n{self.forward_prefix}\n\t<begin>\n"

                    for message_node in messages:
                        sender_name = message_node.get("sender", {}).get("nickname", "未知用户")
                        raw_content = message_node.get("message") or message_node.get("content", [])
                        full_text += f"{sender_name}: "

                        for seg in raw_content:
                            if isinstance(seg, dict):
                                seg_type = seg.get("type")
                                seg_data = seg.get("data", {})
                                if seg_type == "text":
                                    full_text += seg_data.get("text", "")
                                elif seg_type == "at":
                                    full_text += f"[At: {seg_data.get('qq', '')}]"
                                elif seg_type == "image":
                                    img_url = self._extract_image_url(seg_data)
                                    if img_url:
                                        full_text_ref = [full_text]
                                        await self._process_image(img_url, current_message_content, full_text_ref)
                                        full_text = full_text_ref[0]
                        full_text += "\n"

                    full_text += "\t<end>\n"
                    logger.info("检测到合并转发消息，已处理")
                except Exception as e:
                    logger.error(f"处理合并转发失败: {e}")
                    logger.error(traceback.format_exc())

        # 2. 常规消息
        for comp in event.message_obj.message:
            if isinstance(comp, Plain):
                full_text += comp.text
            elif isinstance(comp, At):
                full_text += f" [At: {comp.name if hasattr(comp, 'name') else comp.qq}]"
            elif isinstance(comp, Image):
                url = self._extract_image_url(comp)
                if url:
                    full_text_ref = [full_text]
                    await self._process_image(url, current_message_content, full_text_ref)
                    full_text = full_text_ref[0]
            elif isinstance(comp, Forward):
                pass

        if full_text:
            current_message_content.append({"type": "text", "text": full_text})

        if current_message_content:
            buf = self.session_chats[event.unified_msg_origin]
            buf.append(current_message_content)
            # 缓存上限：防止从不触发LLM的群无限囤积消息（尤其base64图片模式），
            # 2G小机器的内存经不起这么造。超限丢最老的。
            max_buffer = int(self.get_cfg("max_buffer_messages", 60))
            if len(buf) > max_buffer:
                del buf[:len(buf) - max_buffer]
            logger.debug(f"群聊上下文 | {event.unified_msg_origin} | +{len(current_message_content)}组件 | 缓存{len(buf)}条")

    # ==================== LLM请求处理 ====================

    def _control_conversation_rounds(self, req: ProviderRequest, rounds_limit: int):
        if not req.contexts or rounds_limit <= 0:
            return
        round_ends = []
        for i in range(len(req.contexts) - 1):
            current_role = req.contexts[i].get("role")
            next_role = req.contexts[i + 1].get("role")
            if current_role == "assistant" and next_role in ["user", "system"]:
                round_ends.append(i)
        if req.contexts and req.contexts[-1].get("role") == "assistant":
            round_ends.append(len(req.contexts) - 1)
        if len(round_ends) > rounds_limit:
            keep_start_index = round_ends[-rounds_limit]
            req.contexts = req.contexts[keep_start_index:]

    def _control_image_carry_rounds(self, req: ProviderRequest, image_carry_rounds: int):
        if not req.contexts or image_carry_rounds <= 0:
            return
        round_ends = []
        for i in range(len(req.contexts) - 1):
            current_role = req.contexts[i].get("role")
            next_role = req.contexts[i + 1].get("role")
            if current_role == "assistant" and next_role in ["user", "system"]:
                round_ends.append(i)
        if req.contexts and req.contexts[-1].get("role") == "assistant":
            round_ends.append(len(req.contexts) - 1)
        if len(round_ends) > image_carry_rounds:
            keep_start_index = round_ends[-image_carry_rounds]
            for i, ctx in enumerate(req.contexts):
                if i < keep_start_index and ctx.get("role") == "user":
                    if isinstance(ctx.get("content"), list):
                        new_content = []
                        current_text = None
                        for item in ctx["content"]:
                            if item["type"] == "text":
                                text = item["text"]
                                if text.startswith("["):
                                    if current_text:
                                        new_content.append({"type": "text", "text": current_text})
                                    current_text = text
                                else:
                                    if current_text:
                                        current_text += text
                                    else:
                                        current_text = text
                            elif item["type"] == "image_url":
                                if current_text:
                                    current_text += " [图片]"
                                else:
                                    current_text = " [图片]"
                        if current_text:
                            new_content.append({"type": "text", "text": current_text})
                        ctx["content"] = new_content

    @filter.on_llm_request()
    async def on_req_llm(self, event: AstrMessageEvent, req: ProviderRequest):
        """群聊场景：将session_chats注入LLM请求上下文"""
        if event.unified_msg_origin not in self.session_chats:
            return

        rounds_limit = int(self.get_cfg("conversation_rounds_limit", 10))

        # 清洗先前嵌入的system字段
        req.contexts = [
            ctx for ctx in req.contexts
            if not (ctx.get("role") == "system" and ctx.get("content", "").startswith(CHATROOM_SYSTEM_PROMPT[:30]))
        ]

        self._control_conversation_rounds(req, rounds_limit)
        self._control_image_carry_rounds(req, self.image_carry_rounds)

        # 构建群聊上下文
        combined_content = [{"type": "text", "text": CHATROOM_SYSTEM_PROMPT + "\n"}]
        text_prompt_parts = []

        for message in self.session_chats[event.unified_msg_origin]:
            combined_content.extend(message)
            text_part = ""
            for comp in message:
                if comp["type"] == "text":
                    text_part += comp["text"]
                elif comp["type"] == "image_url":
                    text_part += " [图片]"
            if text_part.strip():
                text_prompt_parts.append(text_part.strip())

        req.prompt = ""
        if text_prompt_parts:
            req.prompt = "\n---\n".join(text_prompt_parts)

        has_images = any(item.get("type") == "image_url" for item in combined_content)
        if has_images:
            user_message = {"role": "user", "content": combined_content}
        else:
            text_content = "".join(
                item.get("text", "") for item in combined_content if item.get("type") == "text"
            )
            user_message = {"role": "user", "content": text_content}

        req.contexts.append(user_message)
        self.session_chats[event.unified_msg_origin].clear()

    @filter.on_llm_request()
    async def on_req_llm_private(self, event: AstrMessageEvent, req: ProviderRequest):
        """私聊场景的轮数和图片控制"""
        if not (self.enable_private_control and hasattr(event, 'get_message_type') and
                event.get_message_type() == MessageType.FRIEND_MESSAGE):
            return
        self._control_conversation_rounds(req, self.private_conversation_rounds_limit)
        self._control_image_carry_rounds(req, self.private_image_carry_rounds)

    @filter.on_llm_request(priority=-10000)
    async def on_req_llm_clear_prompt(self, event: AstrMessageEvent, req: ProviderRequest):
        """群聊场景：清空prompt防止重复内容"""
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        req.prompt = ""
        if req.contexts:
            req.contexts = [
                ctx for ctx in req.contexts
                if not (ctx.get("role") == "user" and
                       (ctx.get("content") == "" or
                        (isinstance(ctx.get("content"), list) and not ctx.get("content"))))
            ]

    @filter.on_llm_response(priority=-10000)
    async def on_llm_response(self, event: AstrMessageEvent, resp: LLMResponse):
        """群聊场景：[skip]过滤 + 二次清空prompt"""
        if event.get_message_type() != MessageType.GROUP_MESSAGE:
            return
        if resp and resp.completion_text:
            stripped = resp.completion_text.strip()
            if stripped.lower() == "[skip]":
                logger.info("[skip] 模型返回[skip]，已拦截")
                resp.completion_text = ""
                event.stop_event()
                return
        req = event.get_extra("provider_request")
        if req is not None:
            req.prompt = ""

    async def terminate(self):
        logger.info("[celii_gc] 插件已卸载")
