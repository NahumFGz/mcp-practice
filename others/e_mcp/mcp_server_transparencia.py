# mcp_server_transparencia.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TransparenciaGubernamental")


@mcp.tool()
def buscar_contratos_por_ruc(ruc: str) -> dict:
    """
    Busca contratos de un proveedor por su RUC.
    """
    return {
        "ruc": ruc,
        "total_contratos": 42,
        "monto_total": 1250000.50,
        "entidades_top": ["MINSA", "Gobierno Regional de Lima"],
    }


@mcp.tool()
def obtener_asistencias_congresista(nombre: str) -> dict:
    """
    Obtiene las asistencias de un congresista.
    """
    return {
        "nombre": nombre,
        "asistencias": 152,
        "faltas": 17,
        "licencias": 4,
        "asistencia_pct": 88.4,
    }


@mcp.tool()
def buscar_votaciones_por_tema(tema: str) -> list:
    """
    Busca votaciones de un tema.
    """
    return [
        {
            "titulo": "Ley de Reforma del Sistema de Salud Mental",
            "fecha": "2023-10-15",
            "resultado": "Aprobado",
            "votos_a_favor": 87,
            "votos_en_contra": 28,
        },
        {
            "titulo": "Moción para crear comisión investigadora en ESSALUD",
            "fecha": "2023-06-10",
            "resultado": "Rechazado",
            "votos_a_favor": 51,
            "votos_en_contra": 58,
        },
    ]


if __name__ == "__main__":
    # Servidor SSE en http://localhost:8000/sse
    mcp.run(transport="stdio")
