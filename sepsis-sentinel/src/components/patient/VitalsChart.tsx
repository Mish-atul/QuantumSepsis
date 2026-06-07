import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import type { Vital } from "@/types/database";
import { format } from "date-fns";

interface VitalsChartProps {
  vitals: Vital[];
}

export function VitalsChart({ vitals }: VitalsChartProps) {
  const data = vitals.map((v) => ({
    time: format(new Date(v.timestamp), "HH:mm"),
    HR: v.heart_rate,
    MAP: v.map,
    SpO2: v.spo2,
    Temp: v.temperature,
    RR: v.respiratory_rate,
  }));

  if (data.length === 0) {
    return (
      <Card>
        <CardHeader><CardTitle className="text-sm font-mono">Vitals Trend</CardTitle></CardHeader>
        <CardContent><p className="text-muted-foreground text-xs">No data for the last 6 hours</p></CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-mono uppercase tracking-wider text-muted-foreground">
          6-Hour Vitals Trend
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 18%)" />
            <XAxis dataKey="time" stroke="hsl(215, 16%, 55%)" fontSize={10} fontFamily="monospace" />
            <YAxis stroke="hsl(215, 16%, 55%)" fontSize={10} fontFamily="monospace" />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(220, 18%, 10%)",
                border: "1px solid hsl(220, 13%, 18%)",
                borderRadius: "4px",
                fontFamily: "monospace",
                fontSize: "11px",
              }}
            />
            <Legend wrapperStyle={{ fontFamily: "monospace", fontSize: "10px" }} />
            <Line type="monotone" dataKey="HR" stroke="hsl(0, 84%, 60%)" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="MAP" stroke="hsl(199, 89%, 48%)" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="SpO2" stroke="hsl(142, 71%, 45%)" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="Temp" stroke="hsl(38, 92%, 50%)" strokeWidth={1.5} dot={false} />
            <Line type="monotone" dataKey="RR" stroke="hsl(280, 67%, 60%)" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}
