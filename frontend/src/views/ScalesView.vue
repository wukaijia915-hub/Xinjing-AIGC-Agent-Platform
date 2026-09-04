<template>
  <div>
    <h2>心理量表自评</h2>
    <p class="text-muted">标准化心理量表：为 AI 情绪识别提供效度基准，形成"面部+前庭+量表"多模态评估体系</p>

    <div class="card" v-if="!selectedScale">
      <h3>选择量表</h3>
      <div style="display:flex;gap:12px">
        <button v-for="s in scaleList" :key="s.code" class="btn btn-primary" @click="loadScale(s.code)">
          {{ s.name }} ({{ s.code }} · {{ s.question_count }}题)
        </button>
      </div>
    </div>

    <div class="card" v-if="selectedScale && !result">
      <h3>{{ selectedScale.name }} ({{ selectedScale.code }})</h3>
      <p>{{ selectedScale.description }}</p>
      <div style="display:flex;gap:8px;margin-bottom:12px">
        <span v-for="(lbl,i) in selectedScale.options" :key="i" class="opt-tag">
          {{ i+1 }}={{ lbl }}
        </span>
      </div>

      <div v-for="(q,i) in selectedScale.questions" :key="q.id" class="question-row">
        <div class="q-num">{{ q.id }}</div>
        <div class="q-text">{{ q.text }}</div>
        <div class="q-opts">
          <label v-for="(lbl,oi) in selectedScale.options" :key="oi" :class="{ active: answers[i] === oi+1 }">
            <input type="radio" :name="'q'+q.id" :value="oi+1" v-model="answers[i]" />{{ oi+1 }}
          </label>
        </div>
      </div>

      <div style="display:flex;gap:12px;margin-top:16px">
        <select v-model="studentId" style="padding:8px;font-size:14px">
          <option value="0" disabled>选择学生</option>
          <option v-for="s in students" :key="s.id" :value="s.id">{{ s.name }}</option>
        </select>
        <button class="btn btn-primary" @click="submit" :disabled="!studentId || submitting">
          {{ submitting ? '提交中...' : '提交自评' }}
        </button>
        <button class="btn btn-cancel" @click="selectedScale=null;answers=[]">取消</button>
      </div>
    </div>

    <div class="card" v-if="result">
      <h3>测评结果</h3>
      <div class="result-box">
        <div class="result-main">
          <span class="score">{{ result.standard_score }}</span>
          <span class="score-label">标准分</span>
        </div>
        <div class="result-level" :class="result.level">
          等级：{{ levelText(result.level) }}
        </div>
        <div class="result-dims" v-if="Object.keys(result.dimension_scores||{}).length > 0">
          <h4>维度得分</h4>
          <div v-for="(v,k) in result.dimension_scores" :key="k" class="dim-row">
            <span class="dim-name">{{ k }}</span>
            <span class="dim-bar" :style="{width:v*50+'px',background:v>2?'#ef4444':v>1.5?'#f59e0b':'#10b981'}"></span>
            <span>{{ v }}</span>
          </div>
        </div>
      </div>
      <button class="btn btn-primary" @click="selectedScale=null;result=null;answers=[]" style="margin-top:12px">
        再做一次
      </button>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue";
import axios from "axios";

const scaleList = ref([]);
const selectedScale = ref(null);
const students = ref([]);
const answers = ref([]);
const studentId = ref(0);
const result = ref(null);
const submitting = ref(false);

onMounted(async () => {
  const [sl, st] = await Promise.all([
    axios.get("/api/scales/list/all"),
    axios.get("/api/students"),
  ]);
  scaleList.value = sl.data.data;
  students.value = Array.isArray(st.data) ? st.data : (st.data.data || []);
});

async function loadScale(code) {
  const resp = await axios.get(`/api/scales/${code}`);
  selectedScale.value = resp.data.data;
  answers.value = new Array(selectedScale.value.questions.length).fill(0);
  result.value = null;
}

function levelText(lv) {
  const map = { normal: "正常", mild: "轻度异常", moderate: "中度异常", severe: "偏重" };
  return map[lv] || lv;
}

async function submit() {
  submitting.value = true;
  try {
    const resp = await axios.post("/api/scales/submit", {
      student_id: studentId.value,
      scale_type: selectedScale.value.code,
      answers: answers.value,
    });
    result.value = resp.data.data;
  } catch (e) {
    alert(e.response?.data?.detail || "提交失败");
  } finally {
    submitting.value = false;
  }
}
</script>

<style scoped>
.card { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.text-muted { color: #9e9e9e; font-size: 13px; }
.btn { padding: 10px 20px; border: none; border-radius: 6px; font-size: 14px; cursor: pointer; }
.btn-primary { background: #4f46e5; color: #fff; }
.btn-cancel { background: #e5e7eb; color: #374151; }
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.opt-tag { padding: 2px 8px; background: #e8eaf6; border-radius: 4px; font-size: 12px; color: #4f46e5; }
.question-row { display: flex; align-items: center; gap: 12px; padding: 12px 0; border-bottom: 1px solid #f0f0f0; }
.q-num { width: 28px; height: 28px; border-radius: 50%; background: #4f46e5; color: #fff; display: flex; align-items: center; justify-content: center; font-size: 12px; flex-shrink: 0; }
.q-text { flex: 1; font-size: 14px; }
.q-opts { display: flex; gap: 8px; }
.q-opts label { cursor: pointer; padding: 2px 6px; border-radius: 4px; font-size: 13px; }
.q-opts label.active { background: #4f46e5; color: #fff; }
.q-opts input { display: none; }
.result-box { background: #fafafa; border-radius: 8px; padding: 20px; text-align: center; }
.result-main .score { font-size: 48px; font-weight: bold; color: #4f46e5; }
.score-label { font-size: 14px; color: #9e9e9e; margin-left: 8px; }
.result-level { margin-top: 8px; padding: 4px 12px; border-radius: 4px; display: inline-block; font-size: 14px; }
.result-level.normal { background: #d1fae5; color: #059669; }
.result-level.mild { background: #fef3c7; color: #d97706; }
.result-level.moderate { background: #ffedd5; color: #ea580c; }
.result-level.severe { background: #fee2e2; color: #dc2626; }
.result-dims { text-align: left; margin-top: 16px; }
.result-dims h4 { margin: 0 0 8px; }
.dim-row { display: flex; align-items: center; gap: 8px; padding: 4px 0; font-size: 13px; }
.dim-name { width: 120px; }
.dim-bar { height: 12px; border-radius: 6px; display: inline-block; min-width: 4px; transition: width 0.3s; }
</style>
